package io.github.torchexpo.test

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import io.github.torchexpo.test.vision.ImageClassification
import io.github.torchexpo.test.vision.ImageSegmentation

class MainActivity : AppCompatActivity() {

    private var modelName: String? = null
    private var taskName: String? = null
    private lateinit var result: String;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelName = intent.getStringExtra("modelName")
        taskName = intent.getStringExtra("taskName")
        if (!modelName.isNullOrEmpty() && !taskName.isNullOrEmpty()) {
            result = when (taskName) {
                "image-classification" -> ImageClassification(this, modelName!!).predict()
                "image-segmentation" -> ImageSegmentation(this, modelName!!).predict()
                else -> "ERROR"
            }
        }
    }

    fun getResult(): String = result
}