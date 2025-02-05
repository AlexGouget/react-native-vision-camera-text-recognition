package com.visioncameratextrecognition

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.HashMap

class VisionCameraTextRecognitionPlugin(proxy: VisionCameraProxy, options: Map<String, Any>?) :
    FrameProcessorPlugin() {

    // Utilisez proxy.context au lieu de proxy.reactContext
    private val context = proxy.context

    private var recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val latinOptions = TextRecognizerOptions.DEFAULT_OPTIONS
    private val chineseOptions = ChineseTextRecognizerOptions.Builder().build()
    private val devanagariOptions = DevanagariTextRecognizerOptions.Builder().build()
    private val japaneseOptions = JapaneseTextRecognizerOptions.Builder().build()
    private val koreanOptions = KoreanTextRecognizerOptions.Builder().build()

    private val roiX = (options?.get("roiX") as? Number)?.toInt()
    private val roiY = (options?.get("roiY") as? Number)?.toInt()
    private val roiWidth = (options?.get("roiWidth") as? Number)?.toInt()
    private val roiHeight = (options?.get("roiHeight") as? Number)?.toInt()
    private var lastProcessedTime: Long = 0

    init {
        val language = options?.get("language").toString()
        recognizer = when (language) {
            "latin" -> TextRecognition.getClient(latinOptions)
            "chinese" -> TextRecognition.getClient(chineseOptions)
            "devanagari" -> TextRecognition.getClient(devanagariOptions)
            "japanese" -> TextRecognition.getClient(japaneseOptions)
            "korean" -> TextRecognition.getClient(koreanOptions)
            else -> TextRecognition.getClient(latinOptions)
        }
    }

    override fun callback(frame: Frame, arguments: Map<String, Any>?): HashMap<String, Any>? {
        val currentTime = System.currentTimeMillis()
      
        if (currentTime - lastProcessedTime < 500) {
            return WritableNativeMap().toHashMap()
        }
        lastProcessedTime = currentTime

        val data = WritableNativeMap()
        val mediaImage: Image = frame.image
        val image =
            InputImage.fromMediaImage(mediaImage, frame.imageProxy.imageInfo.rotationDegrees)

        val task = recognizer.process(image)

        try {
            val text = Tasks.await(task)
            if (text.text.isEmpty()) {
                return WritableNativeMap().toHashMap()
            }
            data.putString("resultText", text.text)
            data.putArray("blocks", getBlocks(text.textBlocks))
            return data.toHashMap()
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }

    }


    /**
     * Applique la rotation sur le Bitmap source.
     */
    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    /**
     * Enregistre le Bitmap dans le dossier "debug_cropped_images" du stockage privé de l'application.
     * Renvoie le chemin absolu vers le fichier ou null en cas d'erreur.
     */
    private fun saveBitmapToFile(bitmap: Bitmap, name:String): String? {
        return try {
            val dir = File(context.getExternalFilesDir(null), "debug_cropped_images")
            if (!dir.exists()) {
                dir.mkdirs()
            }
            val file = File(dir, "${name}_${System.currentTimeMillis()}.png")
            val fos = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            fos.flush()
            fos.close()
            file.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Convertit l'image YUV en Bitmap.
     * Notez que l'ordre des buffers U et V peut varier selon l'appareil.
     */
    private fun yuv420ToBitmap(image: Image): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        // Ici, nous plaçons V avant U
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    // Suite des méthodes pour le mapping des blocs de texte
    companion object {
        fun getBlocks(blocks: MutableList<Text.TextBlock>): WritableNativeArray {
            val blockArray = WritableNativeArray()
            blocks.forEach { block ->
                val blockMap = WritableNativeMap().apply {
                    putString("blockText", block.text)
                    putArray("blockCornerPoints", block.cornerPoints?.let { getCornerPoints(it) })
                    putMap("blockFrame", getFrame(block.boundingBox))
                    putArray("lines", getLines(block.lines))
                }
                blockArray.pushMap(blockMap)
            }
            return blockArray
        }

        private fun getLines(lines: MutableList<Text.Line>): WritableNativeArray {
            val lineArray = WritableNativeArray()
            lines.forEach { line ->
                val lineMap = WritableNativeMap().apply {
                    putString("lineText", line.text)
                    putArray("lineCornerPoints", line.cornerPoints?.let { getCornerPoints(it) })
                    putMap("lineFrame", getFrame(line.boundingBox))
                    putArray(
                        "lineLanguages",
                        WritableNativeArray().apply { pushString(line.recognizedLanguage) }
                    )
                    putArray("elements", getElements(line.elements))
                }
                lineArray.pushMap(lineMap)
            }
            return lineArray
        }

        private fun getElements(elements: MutableList<Text.Element>): WritableNativeArray {
            val elementArray = WritableNativeArray()
            elements.forEach { element ->
                val elementMap = WritableNativeMap().apply {
                    putString("elementText", element.text)
                    putArray("elementCornerPoints", element.cornerPoints?.let { getCornerPoints(it) })
                    putMap("elementFrame", getFrame(element.boundingBox))
                }
                elementArray.pushMap(elementMap)
            }
            return elementArray
        }

        private fun getCornerPoints(points: Array<android.graphics.Point>): WritableNativeArray {
            val cornerPoints = WritableNativeArray()
            points.forEach { point ->
                cornerPoints.pushMap(WritableNativeMap().apply {
                    putInt("x", point.x)
                    putInt("y", point.y)
                })
            }
            return cornerPoints
        }

        private fun getFrame(boundingBox: Rect?): WritableNativeMap {
            return WritableNativeMap().apply {
                boundingBox?.let {
                    putDouble("x", it.exactCenterX().toDouble())
                    putDouble("y", it.exactCenterY().toDouble())
                    putInt("width", it.width())
                    putInt("height", it.height())
                    putInt("boundingCenterX", it.centerX())
                    putInt("boundingCenterY", it.centerY())
                }
            }
        }
    }
}
