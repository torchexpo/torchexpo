package io.github.torchexpo.test

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import io.github.torchexpo.test.vision.ImageClassification

class MainActivity : AppCompatActivity() {

    private var modelName: String? = null
    private lateinit var result: String;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelName = intent.getStringExtra("modelName")
        if (!modelName.isNullOrEmpty()) {
            result = ImageClassification(this, modelName!!).predict()
        }
    }

    fun getResult(): String = result
}