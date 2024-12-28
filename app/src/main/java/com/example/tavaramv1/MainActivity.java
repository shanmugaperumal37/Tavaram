package com.example.tavaramv1;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity {

    // UI components
    private EditText imagePathInput;
    private ImageView selectedImage;
    private TextView diseaseResult, diseaseNote;
    private Button tamilButton, englishButton;
    private Interpreter tflite;

    private String selectedDisease = "";
    private String selectedLanguage = "English"; // Default language

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        imagePathInput = findViewById(R.id.imagePathInput);
        selectedImage = findViewById(R.id.selectedImage);
        diseaseResult = findViewById(R.id.diseaseResult);
        diseaseNote = findViewById(R.id.diseaseNote);
        tamilButton = findViewById(R.id.tamilButton);
        englishButton = findViewById(R.id.englishButton);

        // Load TensorFlow Lite model
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set language buttons
        tamilButton.setOnClickListener(this::onLanguageSelect);
        englishButton.setOnClickListener(this::onLanguageSelect);
    }

    // OnSubmitButtonClick - Triggered when the user submits the image path
    public void onSubmitClick(View view) {
        String imagePath = imagePathInput.getText().toString();

        // Process image and detect disease using TensorFlow Lite model
        String detectedDisease = recognizeDisease(imagePath);

        // Display the result (disease detected)
        diseaseResult.setText(detectedDisease);
        selectedDisease = detectedDisease;

        // Load the selected image into the ImageView
        selectedImage.setImageURI(Uri.parse(imagePath));

        // Initially, load the disease note in English
        loadDiseaseNote();
    }

    // Recognize Disease (Call TensorFlow Lite model)
    private String recognizeDisease(String imagePath) {
        // Load the image and convert it to ByteBuffer format
        Bitmap bitmap = loadImage(imagePath);
        if (bitmap == null) return "Error loading image";

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

        // Create input tensor for the model
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, org.tensorflow.lite.DataType.FLOAT32);
        inputFeature0.loadBuffer(byteBuffer);

        // Run inference with the TensorFlow Lite model
        TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 10}, org.tensorflow.lite.DataType.FLOAT32);
        tflite.run(inputFeature0.getBuffer(), outputFeature0.getBuffer());

        // Get the result (assuming the output is a list of probabilities for 10 diseases)
        float[] output = outputFeature0.getFloatArray();

        // Find the disease with the highest probability
        int maxIndex = getMaxIndex(output);

        // Mapping the index to disease name
        String[] diseaseNames = {
                "Septoria Leaf Spot (Tomato)",
                "Tomato Mosaic Virus (Tomato)",
                "Apple Scab (Apple)",
                "Black Rot (Apple)",
                "Common Rust (Corn)",
                "Northern Leaf Blight (Corn)",
                "Black Rot (Grape)",
                "Esca (Black Measles) (Grape)",
                "Early Blight (Potato)",
                "Late Blight (Potato)"
        };

        return diseaseNames[maxIndex]; // Return disease with the highest probability
    }

    // Helper method to load the image and convert it to Bitmap
    private Bitmap loadImage(String imagePath) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(Uri.parse(imagePath));
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Convert Bitmap to ByteBuffer for TensorFlow Lite model input
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3); // RGB, 224x224
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Normalize pixel values to be between 0 and 1
        for (int i = 0; i < intValues.length; i++) {
            int pixelValue = intValues[i];
            byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f); // Red
            byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);  // Green
            byteBuffer.putFloat((pixelValue & 0xFF) / 255.0f);         // Blue
        }

        return byteBuffer;
    }

    // Find index of maximum value in the output array (i.e., highest probability)
    private int getMaxIndex(float[] output) {
        int maxIndex = 0;
        float maxValue = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    // OnLanguageSelect - Triggered when a user selects language
    public void onLanguageSelect(View view) {
        if (view.getId() == R.id.tamilButton) {
            selectedLanguage = "Tamil";
        } else {
            selectedLanguage = "English";
        }
        loadDiseaseNote(); // Reload disease note in the selected language
    }

    // Load the relevant disease note from the assets folder based on language and disease
    private void loadDiseaseNote() {
        String diseaseFileName = getDiseaseFileName(selectedDisease);
        String diseaseNoteContent = readAssetFile(diseaseFileName);

        if (diseaseNoteContent != null) {
            diseaseNote.setText(diseaseNoteContent);
        } else {
            diseaseNote.setText("Note not available.");
        }
    }

    // Generate the file name based on selected disease and language
    private String getDiseaseFileName(String disease) {
        String formattedDisease = disease.replaceAll("\\s+", "_").replaceAll("[()]", "");
        return formattedDisease + "_" + selectedLanguage + ".txt"; // e.g. "Septoria_Leaf_Spot_Tomato_English.txt"
    }

    // Read the content of a file from the assets folder
    private String readAssetFile(String fileName) {
        try {
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open("Septoria_Leaf_Spot_Tomato_en.txt");
            byte[] buffer = new byte[inputStream.available()];
            inputStream.read(buffer);
            return new String(buffer, StandardCharsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Load TensorFlow Lite model
    private ByteBuffer loadModelFile() throws IOException {
        // Load model file from assets
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
