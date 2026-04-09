use crate::footstep_tracker::FootstepEvent;
use anyhow::{ Context, Result };
use std::net::{ ToSocketAddrs, UdpSocket };

#[cfg(feature = "debug")]
use tracing::{ debug, info };

pub struct UdpSender {
    socket: UdpSocket,
    target: std::net::SocketAddr,
}

impl UdpSender {
    pub fn new(target: &str) -> Result<Self> {
        let normalized = if target.contains(':') {
            target.to_string()
        } else {
            format!("{}:7000", target) // default port when omitted
        };

        let target = normalized
            .to_socket_addrs()
            .context("Failed to resolve UDP target address")?
            .next()
            .context("UDP target resolved to no addresses")?;

        let socket = UdpSocket::bind("0.0.0.0:0").context("Failed to bind UDP socket")?;

        #[cfg(feature = "debug")]
        info!(
            "UDP socket bound to {}",
            socket.local_addr().unwrap_or_else(|_| "unknown".parse().unwrap())
        );

        Ok(Self { socket, target })
    }

    pub fn target(&self) -> std::net::SocketAddr {
        self.target
    }

    pub fn send(&self, event: &FootstepEvent) -> Result<()> {
        let (cdx, cdy) = event.footstep.direction.unwrap_or((0.0, 0.0));
        let mut payload = format!("{:.4} {:.4} {:.4} {:.4} {} {}", event.footstep.x, event.footstep.y, cdx, cdy, event.person_id, event.history.len());
        
        for step in &event.history {
            let (dx, dy) = step.direction.unwrap_or((0.0, 0.0));
            payload.push_str(&format!(" {:.4} {:.4} {:.4} {:.4}", step.x, step.y, dx, dy));
        }
        payload.push('\n');

        #[cfg(feature = "debug")]
        debug!("Sending UDP footstep packet: {:?}", payload);
        self.socket
            .send_to(payload.as_bytes(), self.target)
            .context("Failed to send UDP footstep packet")?;

        Ok(())
    }

    pub fn send_path(&self, person_id: usize, path: &[crate::footstep_tracker::Footstep]) -> Result<()> {
        if path.is_empty() { return Ok(()); }
        
        let last = &path[path.len() - 1];
        let age_secs = last.timestamp.elapsed().as_secs_f32();
        
        let mut payload = format!("MATCH {} {} {:.2}", person_id, path.len(), age_secs);
        
        for step in path {
            let (dx, dy) = step.direction.unwrap_or((0.0, 0.0));
            payload.push_str(&format!(" {:.4} {:.4} {:.4} {:.4}", step.x, step.y, dx, dy));
        }
        payload.push('\n');

        #[cfg(feature = "debug")]
        debug!("Sending UDP path packet: {:?}", payload);
        self.socket
            .send_to(payload.as_bytes(), self.target)
            .context("Failed to send UDP path packet")?;

        Ok(())
    }
}
