import os
from datetime import datetime


def _manage_timed_logs(prefix, keep: int = 10):
    """Clean old OpenAI logs and return the path to a new log.

    Parameters
    ----------
    prefix : str
        The prefix to the log.
    keep : int, default=10
        The number of logs to keep.
    
    Returns
    -------
    mlog_path : str
        The absolute path to the new OpenAI log. Note that the log is not created yet.
    """
    base = os.path.join(os.path.dirname(__file__), "..", "logs")
    if not os.path.exists(base):
        os.makedirs(base)
    dtformat = "%Y-%m-%d-%H.%M.%S.%f"
    fnames = [
        fname for fname in os.listdir(base)
        if fname.startswith(f"{prefix}_")
        and fname.endswith(".log")
        and os.path.isfile(os.path.join(base, fname))
    ]
    subs, sube = len(prefix) + 1, -4  # "{prefix}_" and ".log"
    fnames.sort(key=lambda fname: datetime.strptime(fname[subs:sube], dtformat))
    for fname in fnames[:-keep]:
        os.remove(os.path.join(base, fname))
    return os.path.join(base, f"{prefix}_{datetime.now().strftime(dtformat)}.log")