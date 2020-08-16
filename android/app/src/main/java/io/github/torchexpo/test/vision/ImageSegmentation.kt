package io.github.torchexpo.test.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.core.graphics.set
import io.github.torchexpo.test.TorchExpo
import io.github.torchexpo.test.datasets.ColorPalette
import io.github.torchexpo.test.util.DataUtil
import io.github.torchexpo.test.util.FileUtil
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class ImageSegmentation(context: Context, modelName: String) : TorchExpo {
    private val inputFileName = "image_segmentation.jpg"
    private var bitmap: Bitmap
    private var module: Module

    init {
        bitmap = BitmapFactory.decodeStream(context.assets.open(inputFileName))
        module = Module.load(FileUtil.assetFilePath(context, modelName))
    }

    override fun input(): Tensor = TensorImageUtils.bitmapToFloat32Tensor(
        bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
    )

    override fun output(inputTensor: Tensor): Tensor =
        module.forward(IValue.from(inputTensor)).toTensor()

    override fun predict(): String {
        val values = output(input())
        val shape = values.shape()
        val output = DataUtil.argmax(
            values.dataAsFloatArray, shape[0].toInt(), shape[1].toInt(),
            shape[2].toInt()
        )
        val segmentedBitmap = Bitmap.createBitmap(
            output, bitmap.width, bitmap.height,
            Bitmap.Config.ARGB_8888
        ).copy(Bitmap.Config.ARGB_8888, true)
        var idx = 0
        for (i in 0 until bitmap.height) {
            for (j in 0 until bitmap.width) {
                val color = ColorPalette.COLOURS[output[idx++]]
                segmentedBitmap.set(j, i, Color.rgb(color[0], color[1], color[2]))
            }
        }
        return "DONE"
    }
}