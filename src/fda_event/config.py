"""Study-wide settings, kept in one place so scripts, tests and README agree."""
import os

YEARS = list(range(2015, 2025))  # CDER novel approvals, calendar years 2015-2024

# Event-time windows in trading days relative to day 0.
ESTIMATION_WINDOW = (-250, -30)
MIN_ESTIMATION_OBS = 120
EVENT_WINDOWS = {"[0,+1]": (0, 1), "[-1,+1]": (-1, 1), "[-5,+5]": (-5, 5),
                 "[-20,-2]": (-20, -2), "[+2,+20]": (2, 20)}
PLOT_RANGE = (-20, 20)
# Same-ticker approvals closer than this (trading days) are treated as confounded.
CONFOUND_GAP = 10
SMALL_CAP_USD = 2e9
BENCHMARK = "XBI"
OTHER_INDICES = ["IBB", "SPY"]
PRICE_START = "2013-06-01"
PRICE_END = "2025-06-30"

# SEC asks automated clients to identify themselves.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "biotech-fda-event-study research script lukahere007@users.noreply.github.com",
)
