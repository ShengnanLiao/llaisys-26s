target("llaisys-device-nvidia")
    set_kind("static")

    -- 这个 static CUDA target 会被最终 libllaisys.so 使用，
    -- 因此开启 CUDA device-link。
    set_policy("build.cuda.devlink", true)

    add_deps("llaisys-utils")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("../src/device/nvidia/*.cu")

    -- RTX 2080 Ti
    add_cugencodes("sm_75")

    if is_plat("linux") then
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

        -- 普通 .cu -> .o 阶段：
        -- host code 必须生成 PIC
        add_cuflags(
            "-Xcompiler",
            "-fPIC",
            {force = true}
        )

        -- 关键新增：
        -- CUDA device-link 生成 gpucode.cu.o 时也必须使用 PIC
        add_culdflags(
            "-Xcompiler",
            "-fPIC",
            {force = true}
        )
    end

    on_install(function (target) end)
target_end()
target("llaisys-ops-nvidia")
    set_kind("static")

    set_policy("build.cuda.devlink", true)

    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")

    set_languages("cxx17")

    add_files("../src/ops/*/nvidia/*.cu")

    add_cugencodes("sm_75")

    if is_plat("linux") then
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

    on_install(function (target) end)
target_end()