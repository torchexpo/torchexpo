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
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "resnet18.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testGoogLeNet() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "googlenet.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testMNASNet() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "mnasnet0_5.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testMobileNet() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "mobilenet_v2.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testInception() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "inceptionv3.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testDenseNet() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "densenet121.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testShuffleNet() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "shufflenet_v2_x0_5.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testResNext() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "resnext50_32x4d.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testSqueezeNet() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "squeezenet1_0.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

    @Test
    fun testVGG() {
        intent.putExtra("taskName", "image-classification")
        intent.putExtra("modelName", "vgg11.pt")
        activityScenario = ActivityScenario.launch(intent)
        activityScenario.onActivity {
            assertEquals(imagenetResult, it.getResult())
        }
    }

}