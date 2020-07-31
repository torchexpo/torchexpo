package io.github.torchexpo.test

import org.pytorch.Tensor

interface TorchExpo {

    fun input(): Tensor

    fun output(inputTensor: Tensor): Tensor

    fun predict(): String
}