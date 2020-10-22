// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

// Ono (20201021) ---
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <cstdio>
#include <regex>
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

constexpr char kOutputMultiPalmDetections[] = "multi_palm_detections";
constexpr char kOutputMultiHandLandmarks[] = "multi_hand_landmarks";
constexpr char kOutputMultiPalmRects[] = "multi_palm_rects";
constexpr char kOutputMultiHandRects[] = "multi_hand_rects";
// ---

// Get argument ---
DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");
// ---

::mediapipe::Status RunMPPGraph() {

  // Initialize graph ---
  // Load graph config
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  // initialize
  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  // ---

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  // Get input video ---
  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());
  // ---

  // Initialize output VideoWriter ---
  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }
  // ---

  // Connect OutputStreamPoller with "output_video" on the graph ---
  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  // Ono (20201021) ---
  // Connect "poller_detections" with "output_multi_palm_detections" on the graph
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_detections,
                   graph.AddOutputStreamPoller(kOutputMultiPalmDetections));
  // Connect "poller_landmarks" with "output_multi_hand_landmarks" on the graph
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks,
                   graph.AddOutputStreamPoller(kOutputMultiHandLandmarks));
  // Connect "poller_palm_rects" with "output_multi_palm_rects" on the graph
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_palm_rects,
                   graph.AddOutputStreamPoller(kOutputMultiPalmRects));
  // Connect "poller_hand_rects" with "output_multi_hand_rects_from_landmarks" on the graph
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_hand_rects,
                   graph.AddOutputStreamPoller(kOutputMultiHandRects));
  // ---
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  // ---

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  // Ono (20201021) ---
  int frameNum = 0; // For debug
  // difine output folder path
  std::string output_filename = FLAGS_output_video_path.substr(
    FLAGS_output_video_path.rfind("/") + 1
  );
  std::string output_dirpath = std::string() + "./result/" + output_filename + "/";
  mkdir(output_dirpath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
  // ---

  while (grab_frames) {
    std::cout << "FRAMENUM: " << frameNum << std::endl;

    // Capture opencv camera or video frame.
    std::cout << "Capture image." << std::endl;
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    std::cout << "Convert cv::Mat to ImageFrame." << std::endl;
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Input an ImageFrame to graph (gpu) ---
    // Prepare and add graph input packet.
    std::cout << "Input ImageFrame to Graph." << std::endl;
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return ::mediapipe::OkStatus();
        }));
    // ---

    // Take output image from OutputStreamPoller ---
    // Get the graph result packet, or stop if that fails.
    //mediapipe::Packet packet; // original
    //if (!poller.Next(&packet)) break; // original

    // Ono (20201021) ---
    std::cout << "Take output packet." << std::endl;
    mediapipe::Packet packet; // output_video
    mediapipe::Packet packet_detection;
    mediapipe::Packet packet_landmark;
    mediapipe::Packet packet_palm_rects;
    mediapipe::Packet packet_hand_rects;
    
    std::cout << "output_video" << std::endl;
    if (!poller.Next(&packet)){
      std::cout << "Error: Failed to take output_video." << std::endl;
      break;
    }
    std::cout << "detection" << std::endl;
    if (!poller_detections.Next(&packet_detection)) {
      std::cout << "Error: Failed to take detection." << std::endl;
      break;
    }
    std::cout << "landmark" << std::endl;
    if (!poller_landmarks.Next(&packet_landmark)) {
      std::cout << "Error: Failed to take landmark." << std::endl;
      break;
    }
    std::cout << "palm_rects" << std::endl;
    if (!poller_palm_rects.Next(&packet_palm_rects)) {
      std::cout << "Error: Failed to take palm_rects." << std::endl;
      break;
    }
    std::cout << "hand_rects" << std::endl;
    if (!poller_hand_rects.Next(&packet_hand_rects)) {
      std::cout << "Error: Failed to take hand_rects." << std::endl;
      break;
    }
    std::cout << "DEBUG" << std::endl;
    
    // Get output from packet
    auto &output_detections = packet_detection.Get<std::vector<mediapipe::Detection>>();
    auto &output_landmarks = packet_landmark.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    auto &output_palm_rects = packet_palm_rects.Get<std::vector<mediapipe::NormalizedRect>>();
    auto &output_hand_rects = packet_hand_rects.Get<std::vector<mediapipe::NormalizedRect>>();

    // output file
    std::cout << "Output: DETECTIONS." << std::endl;
    for (int j = 0; j < output_detections.size(); j++)
    {
      std::ostringstream os;
      os << output_dirpath + "/"
         << "FRAMENUM=" << frameNum << "_"
         << "detection_"
         << "j=" << j << ".txt";
      std::ofstream outputfile(os.str());

      std::string serializedStr;
      output_detections[j].SerializeToString(&serializedStr);
      outputfile << serializedStr << std::flush;
    }
    std::cout << "Output: LANDMARK." << std::endl;
    for (int j = 0; j < output_landmarks.size(); j++)
    {
      std::ostringstream os;
      os << output_dirpath + "/"
         << "FRAMENUM=" << frameNum << "_"
         << "landmark_"
         << "j=" << j << ".txt";
      std::ofstream outputfile(os.str());

      std::string serializedStr;
      output_landmarks[j].SerializeToString(&serializedStr);
      outputfile << serializedStr << std::flush;
    }
    std::cout << "Output: PALMRECT." << std::endl;
    for (int j = 0; j < output_palm_rects.size(); j++)
    {
      std::ostringstream os;
      os << output_dirpath + "/"
         << "FRAMENUM=" << frameNum << "_"
         << "palmRect_"
         << "j=" << j << ".txt";
      std::ofstream outputfile(os.str());

      std::string serializedStr;
      output_palm_rects[j].SerializeToString(&serializedStr);
      outputfile << serializedStr << std::flush;
    }
    std::cout << "Output: HANDRECT." << std::endl;
    for (int j = 0; j < output_hand_rects.size(); j++)
    {
      std::ostringstream os;
      os << output_dirpath + "/"
         << "FRAMENUM=" << frameNum << "_"
         << "handRect_"
         << "j=" << j << ".txt";
      std::ofstream outputfile(os.str());

      std::string serializedStr;
      output_hand_rects[j].SerializeToString(&serializedStr);
      outputfile << serializedStr << std::flush;
    }
    // ---
    // ---

    // Convert GpuBuffer to ImageFrame.
    std::cout << "Convert GpuBuffer to ImageFrame." << std::endl;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return ::mediapipe::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    std::cout << "Save output data." << std::endl;
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
    
    frameNum++;
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
