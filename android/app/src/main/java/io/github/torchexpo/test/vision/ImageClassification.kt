package io.github.torchexpo.test.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.github.torchexpo.test.TorchExpo
import io.github.torchexpo.test.datasets.ImageNet
import io.github.torchexpo.test.util.DataUtil
import io.github.torchexpo.test.util.FileUtil
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class ImageClassification(context: Context, modelName: String) : TorchExpo {

    private val inputFileName = "image_classification.jpg"
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
        return DataUtil.top(output(input()), ImageNet.TARGET_CLASSES)
    }
}