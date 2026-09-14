# Keep embedded browser-check JavaScript together.
# ruff: noqa: E501
"""Exercise the offline HTML controls, media, and responsive layout."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts/astra_libero"
)
root = parser.parse_args().root.resolve()
checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1440, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto((root / "index.html").as_uri())
    page.screenshot(path=str(root / "index_preview.png"), full_page=True)
    for report in sorted(root.rglob("report.html")):
        page.goto(report.as_uri())
        page.wait_for_function('document.querySelector("video").readyState >= 1')
        count = page.locator(".call").count()
        assert count > 0
        for i in range(count):
            page.locator(".call").nth(i).click()
            assert f"Call {i + 1}:" in page.locator("#calltitle").inner_text()
            assert page.locator("#params").inner_text().startswith("{")
        page.locator("#prev").click()
        page.locator(".call").first.click()
        page.wait_for_function(
            'Array.from(document.querySelectorAll("#before img,#after img")).every(i=>i.complete&&i.naturalWidth>0)'
        )
        media = page.locator("#video").evaluate(
            "(v)=>({duration:v.duration,width:v.videoWidth,height:v.videoHeight})"
        )
        assert media["width"] == 1152 and media["height"] == 440
        expected = len((report.parent / "observations.jsonl").read_text().splitlines()) / 20
        assert abs(media["duration"] - expected) < 0.1
        if count > 1:
            page.evaluate(
                """()=>{const d=JSON.parse(document.getElementById('data').textContent).decisions;const v=document.querySelector('video');v.currentTime=(d[1].frame_before+1)/20;return v.play();}"""
            )
            page.wait_for_function(
                'document.querySelector("#calltitle").textContent.startsWith("Call 2:")'
            )
            page.locator("#video").evaluate("(v)=>v.pause()")
        page.locator(".call").first.click()
        page.locator("#frame").evaluate('(s)=>{s.value=s.max;s.dispatchEvent(new Event("input"))}')
        page.wait_for_function(
            'Array.from(document.querySelectorAll("#allframes img")).every(i=>i.complete&&i.naturalWidth>0)'
        )
        page.screenshot(path=str(report.parent / "report_preview.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
        page.screenshot(path=str(report.parent / "mobile_preview.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1050})
        checks.append(
            {
                "report": str(report.relative_to(root)),
                "call_controls_checked": count,
                "video": media,
                "local_images_loaded": True,
                "mobile_no_horizontal_overflow": True,
            }
        )
    browser.close()
assert not errors, errors
(root / "browser_validation.json").write_text(
    json.dumps({"checks": checks, "page_errors": errors}, indent=2)
)
print(json.dumps(checks, indent=2))
