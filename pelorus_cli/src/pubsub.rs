//! Pub/sub coordination mode shared by the online, offline and raw commands.

use clap::ValueEnum;

/// How Pelorus pub/sub nodes find the nodes they exchange data with.
///
/// Coordinated and peer topics are separate networks, so this must match the
/// mode of the publishers and subscribers Pelorus talks to. Rat sync always
/// uses the Minot coordinator, whatever this is set to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum PubSubCoordMode {
    /// Route through the Minot coordinator, starting an embedded one if none is reachable.
    Autostart,
    /// Connect to other nodes directly, without a coordinator.
    #[default]
    Peer,
}

impl PubSubCoordMode {
    /// Value of Minot's `wind_ratpub_coord_mode`, so an embedded Minot turbine
    /// publishes in the same mode.
    pub fn minot_turbine_env(self) -> &'static str {
        match self {
            Self::Autostart => "autostart",
            Self::Peer => "peer",
        }
    }
}

#[cfg(feature = "mt_pubsub")]
impl From<PubSubCoordMode> for mt_pubsub::CoordMode {
    fn from(mode: PubSubCoordMode) -> Self {
        match mode {
            PubSubCoordMode::Autostart => mt_pubsub::CoordMode::AutoStart,
            PubSubCoordMode::Peer => mt_pubsub::CoordMode::Peer,
        }
    }
}
