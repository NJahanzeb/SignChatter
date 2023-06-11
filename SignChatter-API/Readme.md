<h1>SignChatter-API</h1><br>

<p>To run these files, follow the steps given below:</p>
<ol>
<li>Setup a virtual environment using  "python -m venv your_venv_name"</li>
<li>Activate the virtual environment using "your_venv_name/bin/activate"
<li>Install the required libraries using requirements.txt file "pip install -r requirements.txt"</li>
<li>Change the paths according to your dirctories (comments mentioned in the code)</li>
<li>Run the directory in your local host using "uvicorn main:app --reload"</li>
</ol>

<p>Details on each file:</p>

<ul>
<li>10C_50F_OG2.h5 is a trained LSTM model capable of recognizing 10 ASL classes, it takes a 2D array of dimensions (50,258) as input. 50 meaning keypoints of 50 frames and 258 meaning 258 keypoints extracted from each of the 50 frames.</li>

<li>20C_50F_128B_20T_sorted_55A_A3_30D.h5 is a trained LSTM model capable of recognizing 20 ASL classes, it takes a 2D array of dimensions (50,258) as input. 50 meaning keypoints of 50 frames and 258 meaning 258 keypoints extracted from each of the 50 frames. 128B means it was trained using a batch size of 128. 20T means the test batch size was 20%. A3 means the architecture used was architecture 3 (see ASL_API_20C.py). 30D means the value for all dropout layers in the model is 0.3 or 30%.</li>

<li>PSL_15C_45F_64B_20T.h5 is a trained LSTM model capable of recognizing 15 Pakistani SL classes, it takes a 2D array of dimensions (45,258) as input. 45 meaning keypoints of 45 frames and 258 meaning 258 keypoints extracted from each of the 45 frames. 64B means it was trained using a batch size of 64. 20T means the test batch size was 20%.</li>

<li>ASL_API_10C.py is the API file that uses the 10C_50F_OG2.h5 model. It first receives a test video from the frontend (mobile app or website), it then extracts 50 equally spaced-apart frames from the video. It then extracts keypoints from all the extracted frames and concatenates them into one big 2D array which is then fed to the model for prediction. The prediction string is then returned to the front-end (app or website).</li>

<li>ASL_API_20C.py is the API file that uses the 20C_50F_128B_20T_sorted_55A_A3_30D.h5 model. It works in exactly the same way as ASL_API_10C.py.</li>

<li>PSL_API.py is the API file that uses the PSL_15C_45F_64B_20T.h5 model. It works in exactly the same way as ASL_API_10C.py.</li>
</ul>