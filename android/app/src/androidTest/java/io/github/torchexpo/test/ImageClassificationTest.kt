package io.github.torchexpo.test

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test

class ImageClassificationTest {

    private lateinit var activityScenario: ActivityScenario<MainActivity>
    private val imagenetResult = "white wolf, Arctic wolf, Canis lupus tundrarum"
    private var intent: Intent =
        Intent(ApplicationProvider.getApplicationContext(), MainActivity::class.java)

    @Test
    fun testResNet18() {
        intent.putExtra("modelName", "resnet18.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testGoogleNet() {
        intent.putExtra("modelName", "googlenet.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }
}