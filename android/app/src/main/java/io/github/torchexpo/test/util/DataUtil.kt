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
    }
}