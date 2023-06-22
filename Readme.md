<h1>SignChatter</h1>

<p>App demo video: https://youtu.be/yTiaujF4e8A  
Backend demo video: https://youtu.be/NV_Mu1_D9Bc</p>

<p>SignChatter is a sign language translation system that generates predictions for videos. It works by taking a 5-10 seconds video of a person performing a single sign as input. The video can be newly recorded or selected from storage. The video is sent to the back-end via an API where the script first extracts multiple frames from it, the script then extracts keypoints from the frames and it then feeds these keypoints to the LSTM model.</p>

<p>The model then generates a translation string which is sent back to the front-end via the API and displayed to the user. The system currently works on American and Pakistani Sign Languages and is available via an Android app and a website.</p>

<ol>
<li>All the individual folders contain their own readme.md files, see them to get detailed information about the application</li>
<li>Copy the app.apk to your android device, install it using your package manager to get the SignChatter mobile application</li>
</ol>

<p><em>Note: if you get an 'Unable to get response' error while using the app or website, it means that the API is not live.</em></p>
