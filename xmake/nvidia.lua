--
-- CUDA-compatible GPU backend
--
-- NVIDIA:
--     nvcc + /usr/local/cuda + sm_75
--
-- Iluvatar:
--     CoreX clang++ + /usr/local/corex + ivcore11
--

local is_iluvatar = has_config("iluvatar")

target("llaisys-device-nvidia")
    set_kind("static")

    add_deps("llaisys-utils")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("../src/device/nvidia/*.cu")

    if is_plat("linux") then

        if is_iluvatar then
            ------------------------------------------------
            -- Iluvatar CoreX
            ------------------------------------------------

            -- 天数下关闭 CUDA device-link
            set_policy("build.cuda.devlink", false)
            set_values("cuda.devlink", false)
            set_values("cuda.rdc", false)

            add_includedirs(
                "/usr/local/corex/include",
                {public = true}
            )

            add_linkdirs(
                "/usr/local/corex/lib64",
                "/usr/local/corex/lib",
                {public = true}
            )

            add_syslinks(
                "cudart",
                {public = true}
            )

            add_cuflags(
                "-x",
                "ivcore",
                "--cuda-path=/usr/local/corex",
                "--cuda-gpu-arch=ivcore11",
                "-std=c++17",
                "-Wno-c++17-extensions",
                "-fPIC",
                {force = true}
            )

        else
            ------------------------------------------------
            -- NVIDIA CUDA
            ------------------------------------------------

            set_policy("build.cuda.devlink", true)

            add_cugencodes("sm_75")

            add_includedirs(
                "/usr/local/cuda/include",
                {public = true}
            )

            add_linkdirs(
                "/usr/local/cuda/lib64",
                {public = true}
            )

            add_syslinks(
                "cudart",
                {public = true}
            )

            add_cuflags(
                "-Xcompiler",
                "-fPIC",
                {force = true}
            )

            add_culdflags(
                "-Xcompiler",
                "-fPIC",
                {force = true}
            )
        end
    end

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")

    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")

    set_languages("cxx17")

    add_files("../src/ops/*/nvidia/*.cu")

    if is_plat("linux") then

        if is_iluvatar then
            ------------------------------------------------
            -- Iluvatar CoreX
            ------------------------------------------------

            -- 天数下关闭 CUDA device-link
            set_policy("build.cuda.devlink", false)
            set_values("cuda.devlink", false)
            set_values("cuda.rdc", false)

            add_includedirs(
                "/usr/local/corex/include",
                {public = true}
            )

            add_linkdirs(
                "/usr/local/corex/lib64",
                "/usr/local/corex/lib",
                {public = true}
            )

            add_syslinks(
                "cudart",
                {public = true}
            )

            add_cuflags(
                "-x",
                "ivcore",
                "--cuda-path=/usr/local/corex",
                "--cuda-gpu-arch=ivcore11",
                "-std=c++17",
                "-Wno-c++17-extensions",
                "-fPIC",
                {force = true}
            )

        else
            ------------------------------------------------
            -- NVIDIA CUDA
            ------------------------------------------------

            set_policy("build.cuda.devlink", true)

            add_cugencodes("sm_75")

            add_includedirs(
                "/usr/local/cuda/include",
                {public = true}
            )

            add_linkdirs(
                "/usr/local/cuda/lib64",
                {public = true}
            )

            add_syslinks(
                "cudart",
                {public = true}
            )

            add_cuflags(
                "-Xcompiler",
                "-fPIC",
                {force = true}
            )

            add_culdflags(
                "-Xcompiler",
                "-fPIC",
                {force = true}
            )
        end
    end

    on_install(function (target) end)
target_end()