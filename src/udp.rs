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
        let payload = format!("{:.4} {:.4}\n", event.footstep.x, event.footstep.y);

        #[cfg(feature = "debug")]
        debug!("Sending UDP footstep packet: {:?}", payload);
        self.socket
            .send_to(payload.as_bytes(), self.target)
            .context("Failed to send UDP footstep packet")?;

        Ok(())
    }
}
