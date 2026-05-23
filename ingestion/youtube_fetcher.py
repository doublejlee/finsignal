from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    full_text = " ".join([entry.text for entry in transcript])
    return full_text

if __name__ == "__main__":
    # Meets Kevin video about stocks - feel free to swap any finance YouTube video ID
    video_id = "placeholder"
    text = get_transcript("Hl8sgbmBF98")
    print(text[:500])