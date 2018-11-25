package org.gameofdimension.bert.predict;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Stopwatch;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.util.concurrent.TimeUnit;

/**
 * @author yzq, yzq@leyantech.com
 * @date 2018-11-25.
 */
public class FeatureExtractor {

  /**
   * main.
   *
   * @param args args
   */
  public static void main(String[] args) throws Exception {
    int seqLength = 128;
    int hiddenSize = 768;
    String modelPath = args[0];
    System.out.println(modelPath);
    try (SavedModelBundle b = SavedModelBundle
        .load(modelPath, "serve")) {
      Session sess = b.session();

      String strUniqueId = "0";
      Tensor<Integer> uniqueIds = Tensors.create(new int[]{Integer.valueOf(strUniqueId)});

      String strInputIds = "101, 14698, 10203, 10372, 27635, 11229, 17279, "
          + "12495, 12166, 33535, 24316, 113, 10151, 53482, 117, 12096, "
          + "10217, 10167, 10417, 10139, 13667, 39140, 16925, 10142, 114,"
          + " 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0";
      Tensor<Integer> inputIds = fromStringToTensor(strInputIds, seqLength);

      String strInputMask = "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
          + " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0";
      Tensor<Integer> inputMask = fromStringToTensor(strInputMask, seqLength);

      String strInputTypeIds = "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
          + " 0, 0, 0, 0, 0, 0, 0, 0";
      Tensor<Integer> inputTypeIds = fromStringToTensor(strInputTypeIds, seqLength);

      Stopwatch stopwatch = Stopwatch.createStarted();
      Tensor out = sess.runner().feed("Placeholder", inputIds).feed("Placeholder_1", inputMask)
          .feed("Placeholder_2", inputTypeIds)
          .fetch("bert/encoder/Reshape_13")
          .run().get(0);
      System.out.println(String.format("time cost %d", stopwatch.elapsed(TimeUnit.MILLISECONDS)));

      float[][][] outArr = new float[1][seqLength][hiddenSize];
      out.copyTo(outArr);
      for (int i = 0; i < hiddenSize; i++) {
        System.out.println(String.format("%.4f", outArr[0][5][i]));
      }
    }
  }

  private static Tensor<Integer> fromStringToTensor(String input, int length) {
    int[] arr = Splitter.on(',')
        .trimResults().omitEmptyStrings().splitToList(input).stream()
        .mapToInt(x -> Integer.valueOf(x))
        .toArray();
    Preconditions.checkArgument(length == arr.length);
    Tensor<Integer> tensor = Tensors.create(new int[][]{arr});
    return tensor;
  }
}
