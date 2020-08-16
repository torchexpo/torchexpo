package io.github.torchexpo.test.util

import org.pytorch.Tensor

class DataUtil {
    companion object {
        fun top(outputTensor: Tensor, targetClasses: Array<String>): String {
            val scores = outputTensor.dataAsFloatArray
            var maxScore = -Float.MAX_VALUE
            var maxScoreIdx = -1
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }
            return targetClasses[maxScoreIdx]
        }

        fun argmax(data: FloatArray, dim: Int, height: Int, width: Int): IntArray {
            val result = IntArray(height * width)
            for (i in 0 until height * width) {
                var maxDim = 0
                var maxVal = data[i]
                for (j in 1 until dim) {
                    if (data[(i + height * width * j)] > maxVal) {
                        maxVal = data[(i + height * width * j)]
                        maxDim = j
                    }
                }
                result[i] = maxDim
            }
            return result
        }
    }
}