from src.backtest.engine import run_backtest, load_settings, segment_summaries
from src.html_report import write_report
import os

summary, df = run_backtest()
cfg = load_settings()
out = cfg["paths"]["results_dir"]
write_report(summary, os.path.join(out, "report.html"))

seg, conf = segment_summaries(df)
seg_path = os.path.join(out, "segments_sport_market.csv")
seg.to_csv(seg_path, index=False)
if conf is not None:
    conf.to_csv(os.path.join(out, "segments_confidence.csv"), index=False)

print("Summary:", summary)
print("Segments written:", seg_path)
