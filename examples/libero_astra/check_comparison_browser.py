"""Check the portable comparison's embedded videos, call controls, and totals."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts/libero_model_comparison",
    )
    root = parser.parse_args().root.resolve()
    data = json.loads((root / "comparison.json").read_text())
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "index.html").as_uri())
        assert page.locator("#totals tr").count() == 2
        assert page.locator("#tasks tr").count() == 6
        baseline = page.locator("#totals").inner_text()
        page.locator("#finishing").check()
        assert page.locator("#totals").inner_text() != baseline
        page.locator("#finishing").uncheck()
        assert page.locator("#totals").inner_text() == baseline
        for i, task in enumerate(data["tasks"]):
            page.locator("#task").select_option(str(i))
            page.wait_for_function(
                "Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=1)"
            )
            media = page.locator("video").evaluate_all(
                "vs=>vs.map(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}))"
            )
            assert all(m["width"] == 1152 and m["duration"] > 4 for m in media)
            for side in ("astra", "luna"):
                details = page.locator(f"#{side}-details").locator("..")
                if details.get_attribute("open") is None:
                    details.locator("summary").click()
                expected = len(task[side]["calls"]) + sum(
                    len(phase["calls"]) for phase in task[side]["finishing"]
                )
                assert page.locator(f"#{side}-call option").count() == expected
                for call in sorted({0, expected // 2, expected - 1}):
                    page.locator(f"#{side}-call").select_option(str(call))
                    selected = json.loads(page.locator(f"#{side}-details").inner_text())
                    assert selected["api_latency_s"] > 0
                    assert selected["output_tokens"] > 0
                    records = task[side]["calls"] + [
                        c for phase in task[side]["finishing"] for c in phase["calls"]
                    ]
                    assert selected == records[call]
                details.locator("summary").click()
                assert (
                    page.locator(f"#{side}-download")
                    .get_attribute("href")
                    .startswith("data:video/mp4;base64,")
                )
                name = page.locator(f"#{side}-download").get_attribute("download")
                assert task[side]["model"] in name and name.endswith(".mp4")
            page.locator("#both").click()
            page.wait_for_function(
                "Array.from(document.querySelectorAll('video')).every(v=>v.currentTime>0.1)"
            )
            page.locator("#pause").click()
            assert page.locator("video").evaluate_all("vs=>vs.every(v=>v.paused)")
            checks.append({"task_id": task["task_id"], "videos": media, "controls": "passed"})
        page.screenshot(path=str(root / "comparison_preview.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
        page.screenshot(path=str(root / "mobile_preview.png"), full_page=True)
        browser.close()
    assert not errors, errors
    result = {"tasks": checks, "page_errors": errors, "mobile_no_overflow": True}
    (root / "browser_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
