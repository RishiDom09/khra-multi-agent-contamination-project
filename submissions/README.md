# Run submissions

One folder per completed run, named after the run folder it came from
(`<YYYY-MM-DD>_<model>`), containing exactly two files copied from your local
run — both small and secret-free:

    submissions/2026-07-25_qwen/
        results.json        <- from results/<run>/results.json
        run_metadata.json   <- from logs/<run>/run_metadata.json

That's what the cross-model/cross-architecture comparison tables are built
from. Full transcript folders are NOT committed — zip them onto a GitHub Release
instead.

The `2026-07-25_qwen` folder here is the worked example: Rishi's Groq/Qwen
pilot (Stages 1-3).
