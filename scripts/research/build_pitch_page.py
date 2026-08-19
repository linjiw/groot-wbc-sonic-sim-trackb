"""Assemble the pitch page, embedding the matched 2x2 videos as data URIs."""
import base64, pathlib, sys

O = pathlib.Path('/home/robotixx/.claude/jobs/456fa63a/tmp/pitch')
OUT = pathlib.Path('/home/robotixx/GR00T-WholeBodyControl/docs/progress/sweepcf_pitch.html')

def uri(name):
    data = (O / name).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(data).decode()

VID = {k: uri(f"duck003_{k}.mp4") for k in
       ("nominal_easy", "adapted_easy", "nominal_hard", "adapted_hard")}

HTML = pathlib.Path(sys.argv[1]).read_text()
for key, value in VID.items():
    HTML = HTML.replace(f"@@{key}@@", value)
OUT.write_text(HTML)
mb = OUT.stat().st_size / 1e6
print(f"{mb:.2f} MB -> {OUT}")
if mb > 15:
    raise SystemExit("too large for an artifact")
