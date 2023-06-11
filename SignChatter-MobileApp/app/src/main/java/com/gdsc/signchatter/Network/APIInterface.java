package com.gdsc.signchatter.Network;

import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface APIInterface {
    @Multipart
    @POST("/")
    Call<PredictionModel> uploadVideo(@Part("file\"; filename=\"video.mp4\"") RequestBody video);
}
