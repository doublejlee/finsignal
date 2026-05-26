from youtube_transcript_api import YouTubeTranscriptApi
import browser_cookie3

def get_transcript(video_id):
    try:
        cookies = browser_cookie3.chrome(domain_name='.youtube.com')
        ytt = YouTubeTranscriptApi(cookies=cookies)
    except Exception:
        # Fall back to unauthenticated if cookies fail
        ytt = YouTubeTranscriptApi()
    
    transcript = ytt.fetch(video_id)
    full_text = " ".join([entry.text for entry in transcript])
    return full_text

if __name__ == "__main__":
    text = get_transcript("Hl8sgbmBF98")
    print(text[:500])