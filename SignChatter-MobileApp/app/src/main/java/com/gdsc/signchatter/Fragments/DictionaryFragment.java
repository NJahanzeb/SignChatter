package com.gdsc.signchatter.Fragments;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.gdsc.signchatter.databinding.FragmentDictionaryBinding;

import java.util.ArrayList;

public class DictionaryFragment extends Fragment {

    private FragmentDictionaryBinding binding;
    private ArrayList<String> links;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        binding = FragmentDictionaryBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        ArrayAdapter<String> itemsAdapter = new ArrayAdapter<>(getContext(), android.R.layout.simple_list_item_1, itemsList());
        binding.list.setAdapter(itemsAdapter);
        linksList();
        clickListener();
    }
    public void clickListener(){
        binding.list.setOnItemClickListener((parent, view, position, id) -> launchVideo(position));
    }
    private void launchVideo(int position) {
        String videoUrl = links.get(position);
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(videoUrl));
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        intent.setPackage("com.google.android.youtube");
        startActivity(intent);
    }
    private ArrayList<String> itemsList(){
        ArrayList<String> wordsList = new ArrayList<>();
        wordsList.add("about");
        wordsList.add("accident");
        wordsList.add("africa");
        wordsList.add("afternoon");
        wordsList.add("again");
        wordsList.add("all");
        wordsList.add("always");
        wordsList.add("animal");
        wordsList.add("any");
        wordsList.add("apple");
        wordsList.add("approve");
        wordsList.add("argue");
        wordsList.add("arrive");
        wordsList.add("aunt");
        wordsList.add("baby");
        wordsList.add("back");
        wordsList.add("bake");
        wordsList.add("balance");
        wordsList.add("bald");
        wordsList.add("ball");
        wordsList.add("Banana");
        wordsList.add("Bar");
        wordsList.add("Basement");
        wordsList.add("Basketball");
        wordsList.add("Bath");
        wordsList.add("Bathroom");
        wordsList.add("Bear");
        wordsList.add("Beard");
        wordsList.add("Bed");
        wordsList.add("Bedroom");
        return wordsList;
    }
    private void linksList(){
        links = new ArrayList<>();
        links.add("https://youtu.be/ZUeY_j-dvh4?t=96"); //about
        links.add("https://www.youtube.com/watch?v=IYj4JGrhZ8o&ab_channel=ASLTeachingResources"); //accident
        links.add("https://www.youtube.com/watch?v=rftlYeAql-k&ab_channel=ASLResource"); //africa
        links.add("https://www.youtube.com/watch?v=rftlYeAql-k&ab_channel=ASLResource"); //africa
        links.add("https://www.youtube.com/watch?v=QuafKMvHhsM&ab_channel=Signs"); //afternoon
        links.add("https://www.youtube.com/watch?v=VAQRVGl4b_I&ab_channel=OurBergLife"); //again
        links.add("https://www.youtube.com/watch?v=ETXQfJc3Xbo&ab_channel=ASLTeachingResources"); //all
        links.add("https://www.youtube.com/watch?v=bXZfp2tu5XI&ab_channel=Rachel%27sASLClass"); //always
        links.add("https://www.youtube.com/watch?v=U2hntHWVH5c&ab_channel=OneFactASL"); //animal
        links.add("https://www.youtube.com/watch?v=HP3sljviAxg&ab_channel=SignCueQuestWordOfTheDay"); //any
        links.add("https://www.youtube.com/watch?v=3wIiujOP6Ag&ab_channel=Signs"); //apple
        links.add("https://www.youtube.com/watch?v=pHBb2dnUoEY&ab_channel=Signs"); //approve
        links.add("https://www.youtube.com/watch?v=ydnYnugXZa4&ab_channel=Signs"); //argue
        links.add("https://www.youtube.com/watch?v=kgsXlwRNM6c&ab_channel=Signs"); //arrive
        links.add("https://www.youtube.com/watch?v=_c8v25YlVyY&ab_channel=Signs"); //aunt
        links.add("https://www.youtube.com/watch?v=ljYpx7ee9zg&ab_channel=Signs"); //baby
        links.add("https://youtu.be/rj3Kemk_uj0?t=65"); //back
        links.add("https://www.youtube.com/watch?v=Q0vD9t1u5bI&ab_channel=Signs"); //bake
        links.add("https://youtu.be/-8AWog889YQ"); //balance
        links.add("https://youtu.be/h81FEVG0nKg"); //bald
        links.add("https://www.youtube.com/watch?v=BPb8K7YpJBs&pp=ygUbImJhbGwiIGluIHNpZ24gbGFuZ3VhZ2UgYXNs"); //ball
        links.add("https://www.youtube.com/watch?v=pGz34K-z9rM&pp=ygUdImJhbmFuYSIgaW4gc2lnbiBsYW5ndWFnZSBhc2w%3D");
        links.add("https://www.youtube.com/watch?v=9Zli7o-T3CU&pp=ygUaImJhciIgaW4gc2lnbiBsYW5ndWFnZSBhc2w%3D");
        links.add("https://youtu.be/C_TxYVsGywI");
        links.add("https://youtu.be/hMTxTnXCzOo");
        links.add("https://www.youtube.com/watch?v=r8wxFtDjU5Y");
        links.add("https://youtu.be/GSRwDBAFh70");
        links.add("https://youtu.be/aVfJRXtyS7U");
        links.add("https://www.youtube.com/watch?v=9wdWQUaR0lE");
        links.add("https://youtu.be/6lQwfs5E7lM");
        links.add("https://youtu.be/arBQREsWlUw");
    }
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
        links = null;
    }
}