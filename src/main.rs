mod pose_detector;
mod visualization;
mod footstep_tracker;
mod person_detector;
mod udp;
mod osc;

use anyhow::{ Context, Result };
use clap::Parser;
use opencv::{ highgui, prelude::*, videoio };
use rayon::prelude::*;
use pose_detector::{ PoseDetector, PoseDetectorConfig, Keypoints };
use visualization::{ draw_footsteps, draw_archived_footsteps, draw_bounding_boxes };
use footstep_tracker::FootstepTracker;
use person_detector::{ YoloDetector, YoloDetectorConfig, BoundingBox, PersonTracker };
use std::sync::{ Arc, Mutex };
use udp::UdpSender;
use osc::OscSender;
#[cfg(feature = "debug")]
use std::fs;
#[cfg(feature = "debug")]
use std::path::{ Path, PathBuf };
#[cfg(feature = "debug")]
use std::sync::OnceLock;
#[cfg(feature = "debug")]
use std::time::SystemTime;

#[cfg(feature = "debug")]
use tracing::{ debug, error, info, warn };
#[cfg(feature = "debug")]
use tracing_subscriber::EnvFilter;
#[cfg(feature = "debug")]
use indicatif::{ ProgressBar, ProgressStyle };
#[cfg(feature = "debug")]
use tracing_appender::rolling;
#[cfg(feature = "debug")]
use tracing_subscriber::prelude::*;

#[cfg(feature = "debug")]
const LOG_DIR: &str = "logs";
#[cfg(feature = "debug")]
const LOG_PREFIX: &str = "footstep-tracker.log";
#[cfg(feature = "debug")]
const MAX_LOG_FILES: usize = 168; // 7 days of hourly logs
#[cfg(feature = "debug")]
const MAX_LOG_TOTAL_BYTES: u64 = 200 * 1024 * 1024;

#[cfg(feature = "debug")]
static LOG_GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

#[cfg(feature = "debug")]
fn enforce_log_retention(
    log_dir: &Path,
    prefix: &str,
    max_files: usize,
    max_total_bytes: u64
) -> std::io::Result<()> {
    let mut files: Vec<(PathBuf, SystemTime, u64)> = Vec::new();

    for entry in fs::read_dir(log_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        if !name.starts_with(prefix) {
            continue;
        }

        let meta = fs::metadata(&path)?;
        files.push((
            path,
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            meta.len(),
        ));
    }

    // Newest first; delete from oldest side.
    files.sort_by(|a, b| b.1.cmp(&a.1));

    for (idx, (path, _, _)) in files.iter().enumerate() {
        if idx >= max_files {
            let _ = fs::remove_file(path);
        }
    }

    let mut kept: Vec<(PathBuf, u64)> = files
        .into_iter()
        .take(max_files)
        .map(|(path, _, size)| (path, size))
        .collect();

    let mut total: u64 = kept.iter().map(|(_, size)| *size).sum();
    while total > max_total_bytes && !kept.is_empty() {
        if let Some((path, size)) = kept.pop() {
            let _ = fs::remove_file(path);
            total = total.saturating_sub(size);
        }
    }

    Ok(())
}

#[cfg(feature = "debug")]
fn init_tracing() {
    let _ = fs::create_dir_all(LOG_DIR);
    if let Err(err) = enforce_log_retention(Path::new(LOG_DIR), LOG_PREFIX, MAX_LOG_FILES, MAX_LOG_TOTAL_BYTES) {
        eprintln!("Warning: failed to enforce log retention: {}", err);
    }

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let file_appender = rolling::hourly(LOG_DIR, LOG_PREFIX);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let _ = LOG_GUARD.set(guard);

    let stdout_layer = tracing_subscriber::fmt::layer().with_ansi(true);
    let file_layer = tracing_subscriber::fmt::layer().with_ansi(false).with_writer(non_blocking);

    // ignore errors if already set
    let _ = tracing_subscriber::registry().with(filter).with(stdout_layer).with(file_layer).try_init();
}

#[derive(Clone, Debug)]
enum VideoSource {
    Webcam(i32),
    File(String),
}

#[derive(Parser, Debug)]
#[command(name = "footstep-tracker", about = "Footstep tracking", version, author)]
struct Args {
    /// Path to the CoreML pose detection model package
    #[arg(short, long, value_name = "MODEL_PATH")]
    model_path: Option<String>,

    /// Video source: camera ID (e.g., 0) or video file path
    #[arg(
        default_value = "0",
        value_name = "CAMERA_ID|VIDEO_FILE",
        value_parser = parse_video_source
    )]
    video: VideoSource,

    /// OSC target address for footstep events (host:port)
    #[arg(
        short,
        long,
        value_name = "OSC_TARGET",
        num_args = 0..=1,
        default_missing_value = "127.0.0.1:7001",
    )]
    osc_target: Option<String>,

    /// UDP target address for footstep events (host:port)
    #[arg(
        short = 'u',
        long,
        value_name = "UDP_TARGET",
        num_args = 0..=1,
        default_missing_value = "127.0.0.1:7000"
    )]
    udp_target: Option<String>,

    /// Path to persisted footsteps history JSON file
    #[arg(
        long,
        value_name = "HISTORY_PATH",
        default_value = "data/footstep_history.json"
    )]
    history_path: String,

    /// Disable loading/saving footsteps history to disk
    #[arg(long, default_value_t = false)]
    no_history: bool,
}

fn parse_video_source(input: &str) -> Result<VideoSource, String> {
    if input.contains('.') || input.contains('/') {
        return Ok(VideoSource::File(input.to_string()));
    }

    match input.parse::<i32>() {
        Ok(id) => Ok(VideoSource::Webcam(id)),
        Err(_) => Ok(VideoSource::Webcam(0)),
    }
}

fn main() -> Result<()> {
    #[cfg(feature = "debug")]
    init_tracing();

    let args = Args::parse();
    let model_path = args.model_path.unwrap_or_else(|| "models/rtmpose.mlpackage".to_string());
    let video_source = args.video;
    let history_path = args.history_path;
    let no_history = args.no_history;

    let yolo_config = YoloDetectorConfig::default();
    let mut person_detector = YoloDetector::new(yolo_config).context(
        "Failed to initialize YOLO detector"
    )?;

    let osc_sender = if let Some(osc_target) = args.osc_target {
        let sender = OscSender::new(&osc_target)?;
        #[cfg(feature = "debug")]
        info!("Footstep OSC target: {}", sender.target());
        Some(sender)
    } else {
        #[cfg(feature = "debug")]
        tracing::warn!("No OSC target specified; footstep events will not be sent over OSC");
        None
    };

    let udp_sender = if let Some(udp_target) = args.udp_target {
        let sender = UdpSender::new(&udp_target)?;
        #[cfg(feature = "debug")]
        info!("Footstep UDP target: {}", sender.target());
        Some(sender)
    } else {
        #[cfg(feature = "debug")]
        tracing::warn!("No UDP target specified; footstep events will not be sent over UDP");
        None
    };

    let config = PoseDetectorConfig {
        model_path,
        ..Default::default()
    };

    let worker_count = std::thread
        ::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    #[cfg(feature = "debug")]
    info!("Configuring {} pose detector workers", worker_count);

    #[cfg(feature = "debug")]
    let pb = ProgressBar::new(worker_count as u64).with_style(
        ProgressStyle::default_bar()
            .template("{prefix:.bold.blue} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-")
    );

    #[cfg(feature = "debug")]
    pb.set_prefix("Pose workers");

    let detector_pool: Arc<Mutex<Vec<PoseDetector>>> = Arc::new(
        Mutex::new(
            (0..worker_count)
                .map(|idx| {
                    let res = PoseDetector::new(config.clone());
                    #[cfg(feature = "debug")]
                    {
                        pb.inc(1);
                        if idx + 1 == worker_count {
                            pb.finish_with_message("ready");
                        }
                    }
                    res
                })
                .collect::<Result<Vec<_>>>()
                .context("Failed to initialize pose detector pool")?
        )
    );

    let mut footstep_tracker = FootstepTracker::new(1000);

    if !no_history {
        match footstep_tracker.load_past_histories(&history_path) {
            Ok(loaded) => {
                #[cfg(feature = "debug")]
                info!("Loaded {} historical paths from {}", loaded, history_path);
            }
            Err(err) => {
                #[cfg(feature = "debug")]
                error!("Failed to load history file {}: {:?}", history_path, err);
                #[cfg(not(feature = "debug"))]
                eprintln!("Warning: failed to load history file {}: {}", history_path, err);
            }
        }
    }

    let mut last_saved_history_count = footstep_tracker.past_history_count();

    let mut id_tracker = PersonTracker::new(1000);

    let (mut cap, video_fps) = match &video_source {
        VideoSource::Webcam(camera_id) => {
            #[cfg(feature = "debug")]
            info!("Opening camera {}...", camera_id);
            let cap = videoio::VideoCapture
                ::new(*camera_id, videoio::CAP_AVFOUNDATION)
                .context("Failed to open video capture")?;

            if !videoio::VideoCapture::is_opened(&cap)? {
                #[cfg(feature = "debug")]
                {
                    error!("Error: Could not open camera {}", camera_id);
                    error!(
                        "Try a different camera ID with: cargo run --release -- <model_path> <camera_id>"
                    );
                }
                return Ok(());
            }
            (cap, None)
        }
        VideoSource::File(path) => {
            #[cfg(feature = "debug")]
            info!("Opening video file: {}", path);
            let cap = videoio::VideoCapture
                ::from_file(path, videoio::CAP_ANY)
                .context("Failed to open video file")?;

            if !videoio::VideoCapture::is_opened(&cap)? {
                #[cfg(feature = "debug")]
                {
                    error!("Error: Could not open video file: {}", path);
                    error!("Make sure the file exists and is a valid video format");
                }
                return Ok(());
            }

            let fps = cap.get(videoio::CAP_PROP_FPS)?;
            let frame_count = cap.get(videoio::CAP_PROP_FRAME_COUNT)?;
            #[cfg(feature = "debug")]
            info!("Video: {:.1} FPS, {} frames", fps, frame_count as i32);
            (cap, Some(fps))
        }
    };

    let is_video_file = matches!(video_source, VideoSource::File(_));

    let frame_delay = if let Some(fps) = video_fps {
        if fps > 0.0 {
            (1000.0 / fps) as i32
        } else {
            30 // fallback to 30 FPS
        }
    } else {
        1 // minimal delay for webcam
    };

    #[cfg(feature = "debug")]
    info!("Press 'q' to quit");

    let window_name = "Footstep Tracker - CoreML";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let mut frame_index: usize = 0;

    loop {
        let mut frame = Mat::default();
        if let Err(err) = cap.read(&mut frame) {
            #[cfg(feature = "debug")]
            error!("Video read failed: {:?}", err);
            continue;
        }

        if frame.empty() {
            // loop video back to start
            if is_video_file {
                #[cfg(feature = "debug")]
                debug!("Looping video...");
                if let Err(err) = cap.set(videoio::CAP_PROP_POS_FRAMES, 0.0) {
                    #[cfg(feature = "debug")]
                    error!("Failed to seek video to start: {:?}", err);
                    continue;
                }
                if let Err(err) = cap.read(&mut frame) {
                    #[cfg(feature = "debug")]
                    error!("Failed to read frame after looping video: {:?}", err);
                    continue;
                }

                // if still no frame then stop
                if frame.empty() {
                    break;
                }
            } else {
                break;
            }
        }

        #[cfg(feature = "debug")]
        let start = std::time::Instant::now();

        let people = match person_detector.detect_people(&frame) {
            Ok(people) => people,
            Err(err) => {
                #[cfg(feature = "debug")]
                error!("YOLO detection failed: {:?}", err);
                continue;
            }
        };
        let people_with_ids = id_tracker.assign_ids(people);

        #[cfg(feature = "debug")]
        {
            let yolo_time = start.elapsed();
            if frame_index % 30 == 0 {
                debug!("YOLO detection time: {:.2?}", yolo_time);
            }
        }

        let pose_config = config.clone();
        let detector_pool = detector_pool.clone();

        let processed: Vec<(BoundingBox, Option<(usize, Keypoints)>)> = match people_with_ids
            .par_iter()
            .enumerate()
            .map(
                |(person_idx, (person_id, bbox))| -> Result<_> {
                    let expanded_bbox = bbox.expand(1.4);

                    let person_crop = YoloDetector::crop_region(
                        &frame,
                        &expanded_bbox,
                        pose_config.input_height
                    )?;

                    let mut detector = match (
                        {
                            let mut pool = detector_pool.lock().unwrap();
                            pool.pop()
                        }
                    ) {
                        Some(det) => det,
                        None =>
                            PoseDetector::new(pose_config.clone()).context(
                                "Failed to initialize pose detector for worker"
                            )?,
                    };

                    let mut keyed = None;
                    if let Ok(mut person_keypoints_batch) = detector.detect_pose(&person_crop) {
                        if let Some(person_keypoints) = person_keypoints_batch.first_mut() {
                            let (x, y, w, h) = expanded_bbox.to_pixels(frame.cols(), frame.rows());

                            for kp in person_keypoints.iter_mut() {
                                kp[1] = (kp[1] * (w as f32) + (x as f32)) / (frame.cols() as f32);
                                kp[0] = (kp[0] * (h as f32) + (y as f32)) / (frame.rows() as f32);
                            }

                            keyed = Some((*person_id, person_keypoints.clone()));
                        }

                        #[cfg(feature = "debug")]
                        {
                            if frame_index % 30 == 0 {
                                debug!(
                                    "Person {}: bbox confidence={:.2}",
                                    person_idx,
                                    bbox.confidence
                                );
                            }
                        }
                    }

                    {
                        let mut pool = detector_pool.lock().unwrap();
                        pool.push(detector);
                    }

                    Ok((expanded_bbox, keyed))
                }
            )
            .collect::<Result<Vec<_>>>() {
            Ok(processed) => processed,
            Err(err) => {
                #[cfg(feature = "debug")]
                error!("Pose processing failed: {:?}", err);
                continue;
            }
        };

        let expanded_bboxes: Vec<BoundingBox> = processed
            .iter()
            .map(|(bbox, _)| bbox.clone())
            .collect();

        let keyed_keypoints: Vec<(usize, Keypoints)> = processed
            .into_iter()
            .filter_map(|(_, keyed)| keyed)
            .collect();

        #[cfg(feature = "debug")]
        {
            if frame_index % 30 == 0 {
                let total_time = start.elapsed();
                let fps = 1.0 / total_time.as_secs_f64();

                info!(
                    "Total detection time (YOLO + {} poses): {:.2?} | FPS: {:.1}",
                    people_with_ids.len(),
                    total_time,
                    fps
                );
            }
        }

        let new_footsteps = footstep_tracker.update(&keyed_keypoints);
        let matched_paths = footstep_tracker.get_matched_past_paths();

        if let Some(sender) = osc_sender.as_ref() {
            // First send new footsteps
            for event in &new_footsteps {
                #[cfg(feature = "debug")]
                if let Err(err) = sender.send(event) {
                    error!("Failed to send OSC footstep packet: {:?}", err);
                }

                #[cfg(not(feature = "debug"))]
                {
                    let _ = sender.send(event);
                }
            }
            // Send full match paths
            for (person_id, path) in &matched_paths {
                #[cfg(feature = "debug")]
                if let Err(err) = sender.send_path(*person_id, path) {
                    error!("Failed to send OSC path packet: {:?}", err);
                }

                #[cfg(not(feature = "debug"))]
                {
                    let _ = sender.send_path(*person_id, path);
                }
            }
        }

        if let Some(sender) = udp_sender.as_ref() {
            // First send new footsteps
            for event in &new_footsteps {
                #[cfg(feature = "debug")]
                if let Err(err) = sender.send(event) {
                    error!("Failed to send UDP footstep packet: {:?}", err);
                }

                #[cfg(not(feature = "debug"))]
                {
                    let _ = sender.send(event);
                }
            }
            // Send full match paths
            for (person_id, path) in &matched_paths {
                #[cfg(feature = "debug")]
                if let Err(err) = sender.send_path(*person_id, path) {
                    error!("Failed to send UDP path packet: {:?}", err);
                }

                #[cfg(not(feature = "debug"))]
                {
                    let _ = sender.send_path(*person_id, path);
                }
            }
        }

        if let Err(err) = draw_bounding_boxes(&mut frame, &people_with_ids) {
            #[cfg(feature = "debug")]
            error!("draw_bounding_boxes failed: {:?}", err);
            continue;
        }
        let active_footsteps = footstep_tracker.get_all_footsteps();
        let archived = footstep_tracker.get_archived_footsteps();

        if !no_history {
            let current_history_count = footstep_tracker.past_history_count();
            if current_history_count != last_saved_history_count {
                if let Err(err) = footstep_tracker.save_past_histories(&history_path) {
                    #[cfg(feature = "debug")]
                    error!("Failed to persist history file {}: {:?}", history_path, err);
                    #[cfg(not(feature = "debug"))]
                    eprintln!("Warning: failed to persist history file {}: {}", history_path, err);
                } else {
                    last_saved_history_count = current_history_count;
                    #[cfg(feature = "debug")]
                    debug!(
                        "Persisted {} historical paths to {}",
                        last_saved_history_count,
                        history_path
                    );
                }
            }
        }

        if let Err(err) = draw_footsteps(&mut frame, &active_footsteps) {
            #[cfg(feature = "debug")]
            error!("draw_footsteps failed: {:?}", err);
            continue;
        }

        // show old footsteps only if people are still there to avoid ghosts
        if !active_footsteps.is_empty() {
            if let Err(err) = draw_archived_footsteps(&mut frame, archived) {
                #[cfg(feature = "debug")]
                error!("draw_archived_footsteps failed: {:?}", err);
                continue;
            }
        }

        if let Err(err) = highgui::imshow(window_name, &frame) {
            #[cfg(feature = "debug")]
            error!("imshow failed: {:?}", err);
            continue;
        }

        let pressed = match highgui::wait_key(frame_delay) {
            Ok(key) => key,
            Err(err) => {
                #[cfg(feature = "debug")]
                error!("wait_key failed: {:?}", err);
                continue;
            }
        };

        if pressed == (b'q' as i32) {
            break;
        }

        #[cfg(feature = "debug")]
        if frame_index % 1800 == 0 {
            if
                let Err(err) = enforce_log_retention(
                    Path::new(LOG_DIR),
                    LOG_PREFIX,
                    MAX_LOG_FILES,
                    MAX_LOG_TOTAL_BYTES
                )
            {
                warn!("Log retention pass failed: {}", err);
            }
        }

        frame_index = frame_index.wrapping_add(1);
    }

    if !no_history {
        if let Err(err) = footstep_tracker.save_past_histories(&history_path) {
            #[cfg(feature = "debug")]
            error!("Failed to save history file on shutdown {}: {:?}", history_path, err);
            #[cfg(not(feature = "debug"))]
            eprintln!("Warning: failed to save history file on shutdown {}: {}", history_path, err);
        }
    }

    Ok(())
}
