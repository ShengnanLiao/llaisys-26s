add_rules("mode.debug", "mode.release")
set_encodings("utf-8")
add_includedirs("include")

-- CPU后端
includes("xmake/cpu.lua")

-- CUDA GPU编译开关
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Compile CUDA GPU code")
option_end()

-- CoreX适配开关
option("iluvatar")
    set_default(false)
    set_showmenu(true)
    set_description("Build with Iluvatar CoreX")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    if has_config("iluvatar") then
        add_defines("PLATFORM_ILUVATAR")
    end
    includes("xmake/nvidia.lua")
end

-- 工具类静态库
target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end
    add_files("src/utils/*.cpp")
    on_install(function (target) end)
target_end()

-- 设备管理层静态库
target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils", "llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end
    add_files("src/device/*.cpp")
    on_install(function (target) end)
target_end()

-- 核心逻辑静态库
target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils", "llaisys-device")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end
    add_files("src/core/*/*.cpp")
    on_install(function (target) end)
target_end()

-- Tensor张量静态库
target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end
    add_files("src/tensor/*.cpp")
    on_install(function (target) end)
target_end()

-- 算子运算静态库
target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", {force = true})
    end
    add_files("src/ops/*/*.cpp")
    on_install(function (target) end)
target_end()

-- 最终主动态库
target("llaisys")
    set_kind("shared")

    add_deps(
        "llaisys-utils",
        "llaisys-device",
        "llaisys-core",
        "llaisys-tensor",
        "llaisys-ops"
    )

    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags(
            "-fPIC",
            "-Wno-unknown-pragmas",
            {force = true}
        )
    end

    if has_config("iluvatar") then

        ------------------------------------------------
        -- Iluvatar CoreX
        ------------------------------------------------

        set_policy("build.cuda.devlink", false)
        set_values("cuda.devlink", false)
        set_values("cuda.rdc", false)

        add_linkdirs(
            "/usr/local/corex/lib64",
            "/usr/local/corex/lib",
            {public = true}
        )

        add_links("cudart")

        add_rpathdirs(
            "/usr/local/corex/lib64",
            "/usr/local/corex/lib"
        )

        ------------------------------------------------
        -- 关键：
        -- Xmake CUDA rule 会自动加入 -lcudadevrt，
        -- CoreX 没有这个 NVIDIA device runtime library。
        --
        -- 这里覆盖最终链接过程：
        -- 1. 让 Xmake 正常生成完整链接参数
        -- 2. 删除 -lcudadevrt
        -- 3. 再调用真正 linker
        ------------------------------------------------

        on_link(function (target)

            import("core.tool.linker")

            -- 获取 Xmake 原本准备执行的完整链接命令
            local program, argv = linker.linkargv(
                target:kind(),
                {"cxx"},
                target:objectfiles(),
                target:targetfile(),
                {target = target}
            )

            -- 删除 NVIDIA CUDA device runtime
            local filtered_argv = {}

            for _, arg in ipairs(argv) do
                if arg ~= "-lcudadevrt" and
                   arg ~= "cudadevrt" then
                    table.insert(
                        filtered_argv,
                        arg
                    )
                end
            end

            print(
                "Iluvatar linking without cudadevrt:"
            )

            print(
                program ..
                " " ..
                os.args(filtered_argv)
            )

            -- 确保输出目录存在
            os.mkdir(
                path.directory(
                    target:targetfile()
                )
            )

            -- 真正执行链接
            os.vrunv(
                program,
                filtered_argv
            )
        end)
    end

    add_files(
        "src/llaisys/*.cc"
    )

    set_installdir(".")

    after_install(function (target)

        print(
            "Copying llaisys to python/llaisys/libllaisys/ .."
        )

        if is_plat("windows") then
            os.cp(
                "bin/*.dll",
                "python/llaisys/libllaisys/"
            )
        end

        if is_plat("linux") then
            os.cp(
                "lib/*.so",
                "python/llaisys/libllaisys/"
            )
        end
    end)
target_end()