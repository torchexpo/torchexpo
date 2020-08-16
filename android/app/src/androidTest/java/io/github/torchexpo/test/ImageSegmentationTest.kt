package io.github.torchexpo.test

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test

class ImageSegmentationTest {

    private lateinit var activityScenario: ActivityScenario<MainActivity>
    private val segmentationResult = "DONE"
    private var intent: Intent =
        Intent(ApplicationProvider.getApplicationContext(), MainActivity::class.java)

    @Test
    fun testFCNResNet() {
        intent.putExtra("taskName", "image-segmentation")
        intent.putExtra("modelName", "fcnresnet101.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(segmentationResult, it.getResult())
        }
    }

    @Test
    fun testDeepLabV3ResNet() {
        intent.putExtra("taskName", "image-segmentation")
        intent.putExtra("modelName", "deeplabv3resnet101.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(segmentationResult, it.getResult())
        }
    }

}