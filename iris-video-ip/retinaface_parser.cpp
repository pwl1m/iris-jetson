// Experimental DeepStream parser for InsightFace buffalo_m RetinaFace output.
//
// This library is intentionally not loaded by the production pipeline yet.
// Keep its math aligned with retinaface_decode.py and InsightFace 0.7.3.
#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kFeatureMaps = 3;
constexpr int kStrides[kFeatureMaps] = {8, 16, 32};
constexpr int kAnchors = 2;
constexpr float kNmsThreshold = 0.4F;

struct Candidate {
    float left;
    float top;
    float right;
    float bottom;
    float score;
};

int find_layer(const std::vector<NvDsInferLayerInfo>& layers, const char* name, int fallback) {
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].layerName && std::strcmp(layers[index].layerName, name) == 0) {
            return static_cast<int>(index);
        }
    }
    return fallback < static_cast<int>(layers.size()) ? fallback : -1;
}

bool valid_float_layer(const NvDsInferLayerInfo& layer) {
    // DeepStream 7.1 reports only the per-anchor shape (1/4/10) for these
    // dynamic RetinaFace outputs. The full row count is derived below from
    // the fixed detector input and feature strides.
    return layer.buffer != nullptr && layer.dataType == FLOAT;
}

float intersection_over_union(const Candidate& left, const Candidate& right) {
    const float x1 = std::max(left.left, right.left);
    const float y1 = std::max(left.top, right.top);
    const float x2 = std::min(left.right, right.right);
    const float y2 = std::min(left.bottom, right.bottom);
    const float width = std::max(0.0F, x2 - x1 + 1.0F);
    const float height = std::max(0.0F, y2 - y1 + 1.0F);
    const float overlap = width * height;
    const float area_left = (left.right - left.left + 1.0F) *
                            (left.bottom - left.top + 1.0F);
    const float area_right = (right.right - right.left + 1.0F) *
                             (right.bottom - right.top + 1.0F);
    return overlap / (area_left + area_right - overlap);
}

}  // namespace

extern "C" bool NvDsInferParseRetinaFace(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
    const bool debug = std::getenv("IRIS_RETINAFACE_PARSER_DEBUG") != nullptr;
    if (debug) {
        std::fprintf(stderr, "retinaface parser layers=%zu input=%ux%u\n",
                     outputLayersInfo.size(), networkInfo.width, networkInfo.height);
        for (const auto& layer : outputLayersInfo) {
            std::fprintf(stderr, "  name=%s type=%d dims=%u elements=%u\n",
                         layer.layerName ? layer.layerName : "<null>",
                         static_cast<int>(layer.dataType), layer.inferDims.numDims,
                         layer.inferDims.numElements);
        }
    }
    if (outputLayersInfo.size() < 9 || networkInfo.width == 0 || networkInfo.height == 0 ||
        detectionParams.numClassesConfigured == 0) {
        return false;
    }

    const float threshold = detectionParams.perClassPreclusterThreshold.empty()
                                ? 0.5F
                                : detectionParams.perClassPreclusterThreshold[0];
    if (debug) {
        std::fprintf(stderr, "retinaface parser threshold=%.4f\n", threshold);
    }
    std::vector<Candidate> candidates;

    for (int level = 0; level < kFeatureMaps; ++level) {
        const int grid_width = static_cast<int>(networkInfo.width) / kStrides[level];
        const int grid_height = static_cast<int>(networkInfo.height) / kStrides[level];
        const unsigned int rows = static_cast<unsigned int>(grid_width * grid_height * kAnchors);
        const int score_index = find_layer(outputLayersInfo, (level == 0) ? "446" : (level == 1) ? "466" : "486", level);
        const int box_index = find_layer(outputLayersInfo, (level == 0) ? "449" : (level == 1) ? "469" : "489", level + kFeatureMaps);
        const int kps_index = find_layer(outputLayersInfo, (level == 0) ? "452" : (level == 1) ? "472" : "492", level + 2 * kFeatureMaps);
        if (score_index < 0 || box_index < 0 || kps_index < 0 ||
            !valid_float_layer(outputLayersInfo[score_index]) ||
            !valid_float_layer(outputLayersInfo[box_index]) ||
            !valid_float_layer(outputLayersInfo[kps_index])) {
            if (debug) {
                std::fprintf(stderr, "retinaface parser invalid level=%d score=%d box=%d kps=%d expected=%u\n",
                             level, score_index, box_index, kps_index, rows);
            }
            return false;
        }

        const float* scores = static_cast<const float*>(outputLayersInfo[score_index].buffer);
        const float* boxes = static_cast<const float*>(outputLayersInfo[box_index].buffer);
        if (debug) {
            float maximum = -INFINITY;
            unsigned int selected_count = 0;
            for (unsigned int row = 0; row < rows; ++row) {
                if (std::isfinite(scores[row])) {
                    maximum = std::max(maximum, scores[row]);
                    if (scores[row] >= threshold) {
                        ++selected_count;
                    }
                }
            }
            std::fprintf(stderr, "retinaface parser level=%d max-score=%.6f selected=%u/%u\n",
                         level, maximum, selected_count, rows);
        }
        for (unsigned int row = 0; row < rows; ++row) {
            const float score = scores[row];
            if (!std::isfinite(score) || score < threshold) {
                continue;
            }
            const unsigned int cell = row / kAnchors;
            const float center_x = static_cast<float>(cell % grid_width * kStrides[level]);
            const float center_y = static_cast<float>(cell / grid_width * kStrides[level]);
            const float* distance = boxes + row * 4U;
            Candidate candidate{
                center_x - distance[0] * kStrides[level],
                center_y - distance[1] * kStrides[level],
                center_x + distance[2] * kStrides[level],
                center_y + distance[3] * kStrides[level],
                score,
            };
            if (std::isfinite(candidate.left) && std::isfinite(candidate.top) &&
                std::isfinite(candidate.right) && std::isfinite(candidate.bottom) &&
                candidate.right > candidate.left && candidate.bottom > candidate.top) {
                candidates.push_back(candidate);
            }
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& left, const Candidate& right) {
                         return left.score > right.score;
                     });
    std::vector<Candidate> kept;
    for (const Candidate& candidate : candidates) {
        bool suppressed = false;
        for (const Candidate& selected : kept) {
            if (intersection_over_union(candidate, selected) > kNmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            objectList.push_back({0U, candidate.left, candidate.top,
                                  candidate.right - candidate.left,
                                  candidate.bottom - candidate.top,
                                  candidate.score});
        }
    }
    if (debug) {
        std::fprintf(stderr, "retinaface parser objects=%zu\n", kept.size());
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRetinaFace);
