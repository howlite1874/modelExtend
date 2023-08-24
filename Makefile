# Alternative GNU Make workspace makefile autogenerated by Premake

ifndef config
  config=debug_x64
endif

ifndef verbose
  SILENT = @
endif

ifeq ($(config),debug_x64)
  x_volk_config = debug_x64
  x_vulkan_headers_config = debug_x64
  x_stb_config = debug_x64
  x_glfw_config = debug_x64
  x_vma_config = debug_x64
  x_glm_config = debug_x64
  x_rapidobj_config = debug_x64
  x_tgen_config = debug_x64
  cw2_config = debug_x64
  cw2_shaders_config = debug_x64
  cw2_bake_config = debug_x64
  labutils_config = debug_x64

else ifeq ($(config),release_x64)
  x_volk_config = release_x64
  x_vulkan_headers_config = release_x64
  x_stb_config = release_x64
  x_glfw_config = release_x64
  x_vma_config = release_x64
  x_glm_config = release_x64
  x_rapidobj_config = release_x64
  x_tgen_config = release_x64
  cw2_config = release_x64
  cw2_shaders_config = release_x64
  cw2_bake_config = release_x64
  labutils_config = release_x64

else
  $(error "invalid configuration $(config)")
endif

PROJECTS := x-volk x-vulkan-headers x-stb x-glfw x-vma x-glm x-rapidobj x-tgen cw2 cw2-shaders cw2-bake labutils

.PHONY: all clean help $(PROJECTS) 

all: $(PROJECTS)

x-volk:
ifneq (,$(x_volk_config))
	@echo "==== Building x-volk ($(x_volk_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-volk.make config=$(x_volk_config)
endif

x-vulkan-headers:
ifneq (,$(x_vulkan_headers_config))
	@echo "==== Building x-vulkan-headers ($(x_vulkan_headers_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-vulkan-headers.make config=$(x_vulkan_headers_config)
endif

x-stb:
ifneq (,$(x_stb_config))
	@echo "==== Building x-stb ($(x_stb_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-stb.make config=$(x_stb_config)
endif

x-glfw:
ifneq (,$(x_glfw_config))
	@echo "==== Building x-glfw ($(x_glfw_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-glfw.make config=$(x_glfw_config)
endif

x-vma:
ifneq (,$(x_vma_config))
	@echo "==== Building x-vma ($(x_vma_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-vma.make config=$(x_vma_config)
endif

x-glm:
ifneq (,$(x_glm_config))
	@echo "==== Building x-glm ($(x_glm_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-glm.make config=$(x_glm_config)
endif

x-rapidobj:
ifneq (,$(x_rapidobj_config))
	@echo "==== Building x-rapidobj ($(x_rapidobj_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-rapidobj.make config=$(x_rapidobj_config)
endif

x-tgen:
ifneq (,$(x_tgen_config))
	@echo "==== Building x-tgen ($(x_tgen_config)) ===="
	@${MAKE} --no-print-directory -C third_party -f x-tgen.make config=$(x_tgen_config)
endif

cw2: labutils x-volk x-stb x-glfw x-vma cw2-shaders x-glm x-rapidobj
ifneq (,$(cw2_config))
	@echo "==== Building cw2 ($(cw2_config)) ===="
	@${MAKE} --no-print-directory -C cw2 -f Makefile config=$(cw2_config)
endif

cw2-shaders:
ifneq (,$(cw2_shaders_config))
	@echo "==== Building cw2-shaders ($(cw2_shaders_config)) ===="
	@${MAKE} --no-print-directory -C cw2/shaders -f Makefile config=$(cw2_shaders_config)
endif

cw2-bake: labutils x-tgen x-glm x-rapidobj
ifneq (,$(cw2_bake_config))
	@echo "==== Building cw2-bake ($(cw2_bake_config)) ===="
	@${MAKE} --no-print-directory -C cw2-bake -f Makefile config=$(cw2_bake_config)
endif

labutils:
ifneq (,$(labutils_config))
	@echo "==== Building labutils ($(labutils_config)) ===="
	@${MAKE} --no-print-directory -C labutils -f Makefile config=$(labutils_config)
endif

clean:
	@${MAKE} --no-print-directory -C third_party -f x-volk.make clean
	@${MAKE} --no-print-directory -C third_party -f x-vulkan-headers.make clean
	@${MAKE} --no-print-directory -C third_party -f x-stb.make clean
	@${MAKE} --no-print-directory -C third_party -f x-glfw.make clean
	@${MAKE} --no-print-directory -C third_party -f x-vma.make clean
	@${MAKE} --no-print-directory -C third_party -f x-glm.make clean
	@${MAKE} --no-print-directory -C third_party -f x-rapidobj.make clean
	@${MAKE} --no-print-directory -C third_party -f x-tgen.make clean
	@${MAKE} --no-print-directory -C cw2 -f Makefile clean
	@${MAKE} --no-print-directory -C cw2/shaders -f Makefile clean
	@${MAKE} --no-print-directory -C cw2-bake -f Makefile clean
	@${MAKE} --no-print-directory -C labutils -f Makefile clean

help:
	@echo "Usage: make [config=name] [target]"
	@echo ""
	@echo "CONFIGURATIONS:"
	@echo "  debug_x64"
	@echo "  release_x64"
	@echo ""
	@echo "TARGETS:"
	@echo "   all (default)"
	@echo "   clean"
	@echo "   x-volk"
	@echo "   x-vulkan-headers"
	@echo "   x-stb"
	@echo "   x-glfw"
	@echo "   x-vma"
	@echo "   x-glm"
	@echo "   x-rapidobj"
	@echo "   x-tgen"
	@echo "   cw2"
	@echo "   cw2-shaders"
	@echo "   cw2-bake"
	@echo "   labutils"
	@echo ""
	@echo "For more information, see https://github.com/premake/premake-core/wiki"