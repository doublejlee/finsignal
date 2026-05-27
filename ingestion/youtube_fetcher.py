from youtube_transcript_api import YouTubeTranscriptApi
import browser_cookie3
import http.cookiejar
import requests
from pathlib import Path

COOKIES_FILE = Path(__file__).parent.parent / "cookies.txt"

def _session_from_jar(jar):
    session = requests.Session()
    session.cookies.update(jar)
    return session

def get_transcript(video_id):
    if COOKIES_FILE.exists():
        print("  [auth] using cookies.txt")
        jar = http.cookiejar.MozillaCookieJar()
        jar.load(str(COOKIES_FILE), ignore_discard=True, ignore_expires=True)
        ytt = YouTubeTranscriptApi(http_client=_session_from_jar(jar))
    else:
        ytt = None
        for loader, name in [(browser_cookie3.firefox, "Firefox"), (browser_cookie3.chrome, "Chrome")]:
            try:
                jar = loader(domain_name='.youtube.com')
                ytt = YouTubeTranscriptApi(http_client=_session_from_jar(jar))
                print(f"  [auth] using {name} cookies")
                break
            except Exception:
                continue
        if ytt is None:
            print("  [auth] no auth available, using unauthenticated")
            ytt = YouTubeTranscriptApi()
    
    transcript = ytt.fetch(video_id)
    full_text = " ".join([entry.text for entry in transcript])
    return full_text

if __name__ == "__main__":
    text = get_transcript("Hl8sgbmBF98")
    print(text[:500])