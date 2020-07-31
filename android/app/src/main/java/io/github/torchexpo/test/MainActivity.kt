package io.github.torchexpo.test

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import io.github.torchexpo.test.vision.ImageClassification

class MainActivity : AppCompatActivity() {

    private lateinit var modelName: String
    private lateinit var result: String;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelName = intent.getStringExtra("modelName").toString()
        result = ImageClassification(this, modelName).predict()
    }

    fun getResult(): String = result
}