target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")

    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags(
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-fopenmp",
            "-O3",
            "-march=native",
            {force = true}
        )

        -- OpenMP runtime。
        -- public=true 让最终 shared library 链接时继承这个依赖。
        add_syslinks("gomp", {public = true})
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

