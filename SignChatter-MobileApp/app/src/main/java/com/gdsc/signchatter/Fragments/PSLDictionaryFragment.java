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

public class PSLDictionaryFragment extends Fragment {

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
        intent.setPackage("com.android.chrome");
        startActivity(intent);
    }
    private ArrayList<String> itemsList(){
        ArrayList<String> wordsList = new ArrayList<>();
        wordsList.add("Careful");
        wordsList.add("Dangerous");
        wordsList.add("Excited");
        wordsList.add("Far");
        wordsList.add("Funny");
        wordsList.add("Good");
        wordsList.add("Heavy");
        wordsList.add("Healthy");
        wordsList.add("Important");
        wordsList.add("Intelligent");
        wordsList.add("Interesting");
        wordsList.add("No");
        wordsList.add("Quick");
        wordsList.add("Ready");
        wordsList.add("Yes");


        return wordsList;
    }
    private void linksList(){
        links = new ArrayList<>();
        links.add("https://psl.org.pk/play/dictionary/category/1/5480");
        links.add("https://psl.org.pk/play/dictionary/category/1/5523");
        links.add("https://psl.org.pk/play/dictionary/category/1/5583");
        links.add("https://psl.org.pk/play/dictionary/category/1/5598");
        links.add("https://psl.org.pk/play/dictionary/category/1/5623");
        links.add("https://psl.org.pk/play/dictionary/category/1/5632");
        links.add("https://psl.org.pk/play/dictionary/category/1/5652");
        links.add("https://psl.org.pk/play/dictionary/category/1/5651");
        links.add("https://psl.org.pk/play/dictionary/category/1/5670");
        links.add("https://psl.org.pk/play/dictionary/category/1/5684");
        links.add("https://psl.org.pk/play/dictionary/category/1/5687");
        links.add("https://psl.org.pk/play/dictionary/category/65/4362");
        links.add("https://psl.org.pk/play/dictionary/category/1/5809");
        links.add("https://psl.org.pk/play/dictionary/category/1/5814");
        links.add("https://psl.org.pk/play/dictionary/category/65/4396");
    }
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
        links = null;
    }
}