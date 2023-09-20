#define MYDEBUGS
#define DOUBLETROUBLE
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include "image_conv.h"

inline constexpr int filterWidth = 11;
inline constexpr int halo = filterWidth / 2;

int main(int argc, char* argv[]) {
  const char* inFile = argv[1];
  char* outFile;
  auto inImage = util::read_image(inFile, halo);
  auto outImage = util::allocate_image(inImage.width(), inImage.height(),
                                       inImage.channels());
  auto filter = util::generate_filter(util::filter_type::blur, filterWidth,
                                      inImage.channels());

#define MAXDEVICES 100

  sycl::queue myQueues[MAXDEVICES];
  int howmany_devices = 0;
  try {
    auto P = sycl::platform(sycl::gpu_selector_v);
    auto RootDevices = P.get_devices();

    for (auto &D : RootDevices) {
      myQueues[howmany_devices++] = sycl::queue(D, sycl::property::queue::enable_profiling{});
      if (howmany_devices >= MAXDEVICES)
        break;
    }
  } catch (sycl::exception e) {
    howmany_devices = 1;
    myQueues[0] = sycl::queue(sycl::property::queue::enable_profiling{});
  }

  try {
#ifdef MYDEBUGS
    auto t1 = std::chrono::steady_clock::now();  // Start timing
#endif

#ifdef DOUBLETROUBLE
    sycl::queue myQueue2 = myQueues[(howmany_devices > 1) ? 1 : 0];
    std::cout << "Second queue is running on "
              << myQueue2.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue2.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue2.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo, "\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u", UUID[0], UUID[1], UUID[2], UUID[3], UUID[4], UUID[5], UUID[6], UUID[7], UUID[8], UUID[9], UUID[10], UUID[11], UUID[12], UUID[13], UUID[14], UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif

#ifdef MYDEBUGS
    std::cout << "Running on "
              << myQueues[0].get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueues[0].get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueues[0].get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo, "\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u", UUID[0], UUID[1], UUID[2], UUID[3], UUID[4], UUID[5], UUID[6], UUID[7], UUID[8], UUID[9], UUID[10], UUID[11], UUID[12], UUID[13], UUID[14], UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif

    auto inImgWidth = inImage.width();
    auto inImgHeight = inImage.height();
    auto channels = inImage.channels();
    auto filterWidth = filter.width();
    auto halo = filter.half_width();
    auto globalRange = sycl::range(inImgWidth, inImgHeight / 2);  // Split the work in half
    auto localRange = sycl::range(1, 32);
    auto ndRange = sycl::nd_range(globalRange, localRange);

    auto inBufRange = sycl::range(inImgHeight + (halo * 2), inImgWidth + (halo * 2)) * sycl::range(1, channels);
    auto outBufRange = sycl::range(inImgHeight, inImgWidth) * sycl::range(1, channels);
    auto filterRange = filterWidth * sycl::range(1, channels);

    auto inBuf = sycl::buffer{inImage.data(), inBufRange};
    auto outBuf = sycl::buffer<float, 2>{outBufRange};
    auto filterBuf = sycl::buffer{filter.data(), filterRange};
    outBuf.set_final_data(outImage.data());

    sycl::event e1 = myQueues[0].submit([&](sycl::handler& cgh1) {
      sycl::accessor inAccessor{inBuf, cgh1, sycl::read_only};
      sycl::accessor outAccessor{outBuf, cgh1, sycl::write_only};
      sycl::accessor filterAccessor{filterBuf, cgh1, sycl::read_only};

      cgh1.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
        globalId = sycl::id{globalId[1], globalId[0]};

        auto channelsStride = sycl::range(1, channels);
        auto haloOffset = sycl::id(halo, halo);
        auto src = (globalId + haloOffset) * channelsStride;
        auto dest = globalId * channelsStride;

        float sum[100];  // Assuming channels < 100
        assert(channels < 100);

        for (size_t i = 0; i < channels; ++i) {
          sum[i] = 0.0f;
        }

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset = sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            for (int i = 0; i < channels; ++i) {
              auto channelOffset = sycl::id(0, i);
              sum[i] += inAccessor[srcOffset + channelOffset] * filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        for (size_t i = 0; i < channels; ++i) {
          outAccessor[dest + sycl::id{0, i}] = sum[i];
        }
      });
    });

    // Split the image processing work between the two queues
    sycl::event e2 = myQueues[1].submit([&](sycl::handler& cgh2) {
      auto outAccessor = outBuf.get_access<sycl::access::mode::write>(cgh2);
      cgh2.single_task([=]() {
        int r[2800 + 1];
        int i, k;
        int b, d;
        int c = 0;
	      int hold = 0;

        for (i = 0; i < 2800; i++) {
          r[i] = 2000;
        }
        r[2800] = 0;

        for (k = 2800; k > 0; k -= 14) {
          d = 0;

          i = k;
          for (;;) {
            d += r[i] * 10000;
            b = 2 * i - 1;

            r[i] = d % b;
            d /= b;
            i--;
            if (i == 0) break;
            d *= i;
          }
          outAccessor[hold++] = c + d / 10000;
	        c = d % 10000;
        }
      });
    });

    myQueues[0].wait_and_throw();
    myQueues[1].wait_and_throw();

#ifdef MYDEBUGS
    double time1A = (e1.template get_profiling_info<sycl::info::event_profiling::command_end>() - e1.template get_profiling_info<sycl::info::event_profiling::command_start>());
    auto t2 = std::chrono::steady_clock::now();  // Stop timing
    double time1B = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());

    std::cout << "profiling: Operation completed on device1 in " << time1A << " nanoseconds (" << time1A / 1.0e9 << " seconds)\n";
    std::cout << "chrono: Operation completed on device1 in " << time1B * 1000 << " nanoseconds (" << time1B * 1000 / 1.0e9 << " seconds)\n";
    std::cout << "chrono more than profiling by " << (time1B * 1000 - time1A) << " nanoseconds (" << (time1B * 1000 - time1A) / 1.0e9 << " seconds)\n";

#ifdef DOUBLETROUBLE
    e2.wait();  // Make sure all digits are done being computed
    sycl::host_accessor myD4(outD4);  // The scope of the buffer continues - so we must not use d4[] directly
    std::cout << "First 800 digits of pi: ";
    for (int i = 0; i < 200; ++i) printf("%.4d", myD4[i]);
    std::cout << "\n";

    double time2A = (e2.template get_profiling_info<sycl::info::event_profiling::command_end>() - e2.template get_profiling_info<sycl::info::event_profiling::command_start>());
    std::cout << "profiling: Operation completed on device2 in " << time2A << " nanoseconds (" << time2A / 1.0e9 << " seconds)\n";
#endif
#endif
  } catch (sycl::exception e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outImage, outFile);
}
