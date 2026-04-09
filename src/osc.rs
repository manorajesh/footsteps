use crate::footstep_tracker::FootstepEvent;
use anyhow::{Context, Result};
use std::net::{ToSocketAddrs, UdpSocket};
use rosc::{OscPacket, OscMessage, OscType};
use rosc::encoder;

#[cfg(feature = "debug")]
use tracing::{debug, info};

pub struct OscSender {
    socket: UdpSocket,
    target: std::net::SocketAddr,
}

impl OscSender {
    pub fn new(target: &str) -> Result<Self> {
        let normalized = if target.contains(':') {
            target.to_string()
        } else {
            format!("{}:7001", target) // default OSC port when omitted
        };

        let target = normalized
            .to_socket_addrs()
            .context("Failed to resolve OSC target address")?
            .next()
            .context("OSC target resolved to no addresses")?;

        let socket = UdpSocket::bind("0.0.0.0:0").context("Failed to bind OSC socket")?;

        #[cfg(feature = "debug")]
        info!(
            "OSC socket bound to {}",
            socket.local_addr().unwrap_or_else(|_| "unknown".parse().unwrap())
        );

        Ok(Self { socket, target })
    }

    #[allow(dead_code)]
    pub fn target(&self) -> std::net::SocketAddr {
        self.target
    }

    pub fn send(&self, event: &FootstepEvent) -> Result<()> {
        let (cdx, cdy) = event.footstep.direction.unwrap_or((0.0, 0.0));
        let mut args = vec![
            OscType::Float(event.footstep.x),
            OscType::Float(event.footstep.y),
            OscType::Float(cdx),
            OscType::Float(cdy),
            OscType::Int(event.person_id as i32),
            OscType::Int(event.history.len() as i32),
        ];

        for step in &event.history {
            let (dx, dy) = step.direction.unwrap_or((0.0, 0.0));
            args.push(OscType::Float(step.x));
            args.push(OscType::Float(step.y));
            args.push(OscType::Float(dx));
            args.push(OscType::Float(dy));
        }

        let msg_buf = encoder::encode(&OscPacket::Message(OscMessage {
            addr: "/footstep".to_string(),
            args,
        })).context("Failed to encode OSC packet")?;

        #[cfg(feature = "debug")]
        debug!("Sending OSC footstep packet to {}", self.target);
        self.socket
            .send_to(&msg_buf, self.target)
            .context("Failed to send OSC footstep packet")?;

        Ok(())
    }

    pub fn send_path(&self, person_id: usize, path: &[crate::footstep_tracker::Footstep]) -> Result<()> {
        if path.is_empty() { return Ok(()); }

        let last = &path[path.len() - 1];
        let age_secs = last.timestamp.elapsed().as_secs_f32();

        let (ldx, ldy) = last.direction.unwrap_or((0.0, 0.0));
        let mut args = vec![
            OscType::Float(last.x),
            OscType::Float(last.y),
            OscType::Float(ldx),
            OscType::Float(ldy),
            OscType::Int(person_id as i32),
            OscType::Int(path.len() as i32),
            OscType::Float(age_secs),
        ];

        for step in path {
            let (dx, dy) = step.direction.unwrap_or((0.0, 0.0));
            args.push(OscType::Float(step.x));
            args.push(OscType::Float(step.y));
            args.push(OscType::Float(dx));
            args.push(OscType::Float(dy));
        }

        let msg_buf = encoder::encode(&OscPacket::Message(OscMessage {
            addr: "/match".to_string(),
            args,
        })).context("Failed to encode OSC path packet")?;

        #[cfg(feature = "debug")]
        debug!("Sending OSC path packet to {}", self.target);
        self.socket
            .send_to(&msg_buf, self.target)
            .context("Failed to send OSC path packet")?;

        Ok(())
    }
}
