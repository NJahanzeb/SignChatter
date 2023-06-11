package com.gdsc.signchatter.Fragments;

import android.Manifest;
import android.app.Activity;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.media.CamcorderProfile;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;
import android.speech.tts.TextToSpeech;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.gdsc.signchatter.Network.APIClient;
import com.gdsc.signchatter.Network.APIInterface;
import com.gdsc.signchatter.Network.PredictionModel;
import com.gdsc.signchatter.R;
import com.gdsc.signchatter.databinding.FragmentHomeBinding;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

import okhttp3.MediaType;
import okhttp3.RequestBody;
import pl.droidsonroids.gif.GifImageView;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class HomeFragment extends Fragment {

    private FragmentHomeBinding binding;
    private TextToSpeech textToSpeech;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        binding = FragmentHomeBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        binding.ivCamera.setOnClickListener(v -> launchCamera());
        binding.ivSelect.setOnClickListener(v-> launchGallery());
    }

    private void launchCamera() {
        if (ContextCompat.checkSelfPermission(getContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.CAMERA}, 200);
        } else {
            Intent takeVideoIntent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
            takeVideoIntent.putExtra(MediaStore.EXTRA_DURATION_LIMIT, 8);
            takeVideoIntent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, CamcorderProfile.QUALITY_480P);
            if (takeVideoIntent.resolveActivity(getActivity().getPackageManager()) != null) {
                someActivityResultLauncher.launch(takeVideoIntent);
            }
        }
    }



    public void launchGallery() {
        Intent intent = new Intent();
        intent.setType("video/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        someActivityResultLauncher.launch(intent);
    }

    ActivityResultLauncher<Intent> someActivityResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    Uri uri = data.getData();
                    long durationMillis = getVideoDuration(uri);
                    if (durationMillis < 2*1000 || durationMillis > 8*1000) {
                        Toast.makeText(getContext(), "Video length should be between 2 and 8 seconds", Toast.LENGTH_LONG).show();
                        return;
                    }
                    convertToBytes(uri);
                }
            });

    private long getVideoDuration(Uri uri) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(getContext(), uri);
        String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
        return Long.parseLong(durationStr);
    }

    private void convertToBytes(Uri uri) {

        byte[] videoBytes;
        try {
            videoBytes = getBytes(uri);
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        if(videoBytes!=null){
            callApi(videoBytes);
        }
    }
    private void callApi(byte[] videoBytes) {
        Dialog d = showDialog();
        GifImageView gifImageView = d.findViewById(R.id.ivWait);
        TextView tv = d.findViewById(R.id.tv_result);
        APIInterface APIInterface = APIClient.getClient().create(APIInterface.class);
        RequestBody requestBody = RequestBody.create(videoBytes, MediaType.parse("application/octet-stream"));
        Call<PredictionModel> predictionModelCall = APIInterface.uploadVideo(requestBody);
        predictionModelCall.enqueue(new Callback<>() {
            @Override
            public void onResponse(@NonNull Call<PredictionModel> call, @NonNull Response<PredictionModel> response) {
                if (response.isSuccessful()) {
                    PredictionModel model = response.body();
                    assert model != null;
                    String res = model.Prediction;
                    gifImageView.setImageResource(R.drawable.feedback);
                    tv.setText(res);
                    convertTextToSpeech(res);

                }
            }

            @Override
            public void onFailure(@NonNull Call<PredictionModel> call, @NonNull Throwable t) {
                call.cancel();
                gifImageView.setImageResource(R.drawable.sad);
                tv.setText("Unable to get response");
            }
        });

    }

    private Dialog showDialog() {
        Dialog dialog=new Dialog(getContext());
        dialog.setContentView(R.layout.dialog_style);
        dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.WHITE));
        dialog.setCanceledOnTouchOutside(false);
        dialog.show();
        return dialog;
    }

    public byte[] getBytes(Uri u) throws IOException {
        try {
            InputStream inputStream = getContext().getContentResolver().openInputStream(u);
            ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
            byte[] buffer = new byte[2048];
            int len;
            while ((len = inputStream.read(buffer)) != -1) {
                byteBuffer.write(buffer, 0, len);
            }
            inputStream.close();
            return byteBuffer.toByteArray();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
    private void convertTextToSpeech(String text) {
        if (textToSpeech == null) {
            textToSpeech = new TextToSpeech(requireContext(), new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int status) {
                    if (status == TextToSpeech.SUCCESS) {
                        int result = textToSpeech.setLanguage(Locale.US);
                        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                            Toast.makeText(requireContext(), "Language not supported", Toast.LENGTH_SHORT).show();
                        } else {
                            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null);
                        }
                    } else {
                        Toast.makeText(requireContext(), "Initialization failed", Toast.LENGTH_SHORT).show();
                    }
                }
            });
        } else {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
    }
}







