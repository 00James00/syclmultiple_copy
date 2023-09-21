
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

  if (argc == 2) {
    if (strchr(inFile, '/') || strchr(inFile, '\\')) {
      std::cerr << "Sorry, filename cannot include a path.\n";
      exit(1);
    }
    const char* prefix = "blurred_";
    size_t len1 = strlen(inFile);
    size_t len2 = strlen(prefix);
    outFile = (char*)malloc((len1 + len2 + 1) * sizeof(char));
    strcpy(outFile, prefix);
    strcpy(outFile + 8, inFile);
#ifdef MYDEBUGS
    std::cout << "Input file: " << inFile << "\nOutput file: " << outFile
              << "\n";
#endif
  } else {
    std::cerr << "Usage: " << argv[0] << " imagefile\n";
    exit(1);
  }

  auto inImage = util::read_image(inFile, halo);

  auto outImage = util::allocate_image(inImage.width(), inImage.height(),
                                       inImage.channels());

  // The image convolution support code provides a
  // `filter_type` enum which allows us to choose between
  // `identity` and `blur`. The utility for generating the
  // filter data; `generate_filter` takes a `filter_type`
  // and a width.

  auto filter = util::generate_filter(util::filter_type::blur, filterWidth,
                                      inImage.channels());


  //
  // This code tries to grab up to 100 (MAXDEVICES) GPUs.
  // If there are no GPUs, it will get a default device.
  //
#define MAXDEVICES 100

  sycl::queue myQueues[MAXDEVICES];
  int howmany_devices = 0;
  try {
    auto P = sycl::platform(sycl::gpu_selector_v);
    auto RootDevices = P.get_devices();
    // auto C = sycl::context(RootDevices);
    for (auto &D : RootDevices) {
      myQueues[howmany_devices++] = sycl::queue(D,sycl::property::queue::enable_profiling{});
      if (howmany_devices >= MAXDEVICES)
	break;
    }
  } 
  catch (sycl::exception e) {
    howmany_devices = 1;
    myQueues[0] = sycl::queue(sycl::property::queue::enable_profiling{});
  }

#ifdef DEBUGDUMP
  for (int i = 0; i < howmany_devices; ++i) {
    std::cout << "Device: "
	      << myQueues[i].get_device().get_info<sycl::info::device::name>()
	      << " MaxComputeUnits: " << myQueues[i].get_device().get_info<sycl::info::device::max_compute_units>();
    if (myQueues[i].get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueues[i].get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",
	      UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],
	      UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
    std::cout << "\n";
  }
#endif

  try {
    sycl::queue myQueue1 = myQueues[0];

   
#ifdef MYDEBUGS
    std::cout << "Running on "
              << myQueue1.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue1.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue1.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif

#ifdef DOUBLETROUBLE
    sycl::queue myQueue2 = myQueues[ (howmany_devices > 1) ? 1 : 0 ];
    std::cout << "Second queue is running on "
              << myQueue2.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue2.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue2.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif


#ifdef DOUBLETROUBLE
    std::array<int, 200> d4;
    // inspired and based upon:
    // https://cs.uwaterloo.ca/~alopez-o/math-faq/mathtext/node12.html
    // and
    // https://crypto.stanford.edu/pbc/notes/pi/code.html
    // (retrieved September 13, 2023)
    //
    sycl::buffer outD4(d4);
    sycl::event e2 = myQueue2.submit([&](sycl::handler& cgh2) {
      auto outAccessor = outD4.get_access<sycl::access::mode::write>(cgh2);
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
#endif

  auto inImgWidth = inImage.width();
  auto inImgHeight = inImage.height();
  auto channels = inImage.channels();
  auto filterWidth = filter.width();
  auto halo = filter.half_width();

  auto inImgHeightHalf = inImgWidth / 2;
  auto inImgHeightRem = inImgHeight - inImgHeightHalf;

  auto globalRange = sycl::range(inImgWidth, inImgHeight);
  auto localRange = sycl::range(1, 32);
  auto ndRange = sycl::nd_range(globalRange, localRange);

  auto inBufRange =
      sycl::range(inImgHeight + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
  auto outBufRange =
      sycl::range(inImgHeight, inImgWidth) * sycl::range(1, channels);
  auto inBufRangeHalf = sycl::range(inImgHeightHalf + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
  auto outBufRangeHalf = sycl::range(inImgHeightHalf, inImgWidth) * sycl::range(1, channels);
  auto inBufRangeRem = sycl::range(inImgHeightRem + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
  auto outBufRangeRem = sycl::range(inImgHeightRem, inImgWidth) * sycl::range(1, channels);


  // std::cout << "Buffer range x: " << inBufRange[0] << " y: " << inBufRange[1] << std::endl;
  // std::cout << "Buffer range 1st half x: " << inBufRangeHalf[0] << " y: " << inBufRangeHalf[1] << std::endl;
  // std::cout << "Buffer range 2nd half x: " << inBufRangeRem[0] << " y: " << inBufRangeRem[1] << std::endl;

  auto filterRange = filterWidth * sycl::range(1, channels);

#ifdef MYDEBUGS
  std::cout << "inImgWidth: " << inImgWidth << "\ninImgHeight: " << inImgHeight
            << "\nchannels: " << channels << "\nfilterWidth: " << filterWidth
            << "\nhalo: " << halo << "\n";
#endif

  // Always good to limit scope of accessors,
  // so a good SYCL program will introduce a scope before
  // defining buffers.
  // Remember: While a buffer exists, the data it points
  // to should ONLY be accessed with an accessor. That
  // goes for the host just as much as the device.

  {
    auto inBuf = sycl::buffer{inImage.data(), inBufRange};
    auto outBuf = sycl::buffer<float, 2>{outBufRange};
    auto filterBuf = sycl::buffer{filter.data(), filterRange};
    outBuf.set_final_data(outImage.data());

      // Insert a trivial kernel just before starting the timer
    sycl::event trivialEvent = myQueues[0].submit([&](sycl::handler& cgh) {
      cgh.single_task<class TrivialTask>([=]() {
        // This is another trivial task, you can put any code here
        for (int j = 0; j < 1000000; ++j) {
          // Do some simple computation
          double temp = j * 2.71828;
        }
      });
    });

    trivialEvent.wait(); // Wait for the trivial task to complete
    #ifdef MYDEBUGS
    auto t1 = std::chrono::steady_clock::now();  // Start timing
    #endif


    sycl::event e1 = myQueue1.submit([&](sycl::handler& cgh1) {
      // sycl::accessor inAccessor{inBuf, cgh1, sycl::read_only};
      // sycl::accessor outAccessor{outBuf, cgh1, sycl::write_only};
      auto inAccessor = inBuf.get_access<sycl::access::mode::read>(
        cgh1, sycl::range(522, 1566));
      auto outAccessor = outBuf.get_access<sycl::access::mode::write>(
        cgh1, sycl::range(512, 1536)
      );
      sycl::accessor filterAccessor{filterBuf, cgh1, sycl::read_only};

      cgh1.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
        globalId = sycl::id{globalId[1], globalId[0]};

        auto channelsStride = sycl::range(1, channels);
        auto haloOffset = sycl::id(halo, halo);
        auto src = (globalId + haloOffset) * channelsStride;
        auto dest = globalId * channelsStride;

        // 100 is a hack - so the dim is not dynamic
        float sum[/* channels */ 100];
        assert(channels < 100);

        for (size_t i = 0; i < channels; ++i) {
          sum[i] = 0.0f;
        }
        sum[0] = 0.0f;

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset =
                sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            for (int i = 0; i < channels; ++i) {
              auto channelOffset = sycl::id(0, i);
              sum[i] += inAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        // outAccessor[dest + sycl::id{0, 0}] = sum[0];

        for (size_t i = 0; i < channels; ++i) {
          outAccessor[dest + sycl::id{0, i}] = sum[i];
        }
      });
    });




    sycl::event e3 = myQueue2.submit([&](sycl::handler& cgh3) {
      // sycl::accessor inAccessor{inBuf, cgh1, sycl::read_only};
      // sycl::accessor outAccessor{outBuf, cgh1, sycl::write_only};
      // Allow full access but only write to the 3rd channel?
      auto inAccessor = inBuf.get_access<sycl::access::mode::read>(
        cgh3, sycl::range(522, 1566)
        // sycl::id(0, 1044)
      );
      auto outAccessor = outBuf.get_access<sycl::access::mode::write>(
        cgh3, sycl::range(512, 1536)
        // sycl::id(0, 1024)
      );
      sycl::accessor filterAccessor{filterBuf, cgh3, sycl::read_only};

      cgh3.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
        globalId = sycl::id{globalId[1], globalId[0]};

        auto channelsStride = sycl::range(1, channels);
        auto haloOffset = sycl::id(halo, halo);
        auto src = (globalId + haloOffset) * channelsStride;
        auto dest = globalId * channelsStride;

        // 100 is a hack - so the dim is not dynamic
        float sum[/* channels */ 100];
        assert(channels < 100);

        for (size_t i = 1; i < channels; ++i) {
          sum[i] = 0.0f;
        }

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset =
                sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            // Out of bound write here? Only should write to 3rd channel.
            for (int i = 1; i < channels; ++i) {
              auto channelOffset = sycl::id(0, channels - 1);
              sum[channels - 1] += inAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        outAccessor[dest + sycl::id{0, 1}] = sum[1];
        outAccessor[dest + sycl::id{0, 2}] = sum[2];
        // for (size_t i = 0; i < channels; ++i) {
        //   if (i != channels - 1) {
        //     continue;
        //   }
        //   outAccessor[dest + sycl::id{0, 2}] = sum[2];
        // }
      });
    });

    myQueue1.wait_and_throw();
    auto t2 = std::chrono::steady_clock::now();  // Stop timing

#ifdef MYDEBUGS
    // Timing code is from our book (2nd edition) -
    // read section on profiling in
    // Chapter 13 that includes figures 13-6 through 13-8.
    // Check https://tinyurl.com/reinders-4class for link
    // to copy of 2nd edition ("Learn SYCL").

    double time1A = (e1.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e1.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());
    
    double time1B =
        (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
             .count());

    std::cout << "profiling: Operation completed on device1 in " << time1A
              << " nanoseconds (" << time1A / 1.0e9 << " seconds)\n";
    std::cout << "chrono: Operationd completed on device1 in " << time1B * 1000
              << " nanoseconds (" << time1B * 1000 / 1.0e9 << " seconds)\n";
    std::cout << "chrono more than profiling by " << (time1B * 1000 - time1A)
              << " nanoseconds (" << (time1B * 1000 - time1A) / 1.0e9
              << " seconds)\n";

myQueue2.wait_and_throw();

#ifdef DOUBLETROUBLE
    e2.wait(); // make sure all digits are done being computed
    sycl::host_accessor myD4(outD4); // the scope of the buffer continues - so we must not use d4[] directly
    std::cout << "First 800 digits of pi: ";
    for (int i = 0; i < 200; ++i) printf("%.4d", myD4[i]);
    std::cout << "\n";

    double time2A = (e2.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e2.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());
    std::cout << "profiling: Operation completed on device2 in " << time2A
              << " nanoseconds (" << time2A / 1.0e9 << " seconds)\n";
#endif
#endif
  }
}
catch (sycl::exception e) {
  std::cout << "Exception caught: " << e.what() << std::endl;
}

util::write_image(outImage, outFile);
}
