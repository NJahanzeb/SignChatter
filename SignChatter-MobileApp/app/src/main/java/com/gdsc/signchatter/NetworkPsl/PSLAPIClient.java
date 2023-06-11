package com.gdsc.signchatter.NetworkPsl;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class PSLAPIClient {
    private static Retrofit retrofit = null;

   public static Retrofit getClient() {

        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS).build();

        retrofit = new Retrofit.Builder()
                .baseUrl("http://194.163.183.60:19468/")
                .addConverterFactory(GsonConverterFactory.create())
                .client(client)
                .build();


        return retrofit;
    }
}
