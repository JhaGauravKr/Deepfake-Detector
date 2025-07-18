from django import forms 

class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(
        label="Upload Video File",
        help_text="Max file size: 100MB. Allowed formats: mp4, gif, webm, avi, 3gp, wmv, flv, mkv."
    )
    sequence_length = forms.IntegerField(
        label="Sequence Length",
        min_value=1,
        initial=60,
        help_text="Number of frames to extract for analysis. (e.g., 60)"
    )