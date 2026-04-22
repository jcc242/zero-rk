
add_library(reactor variable_volume_reactor.cpp const_pressure_reactor.cpp counterflow_reactor.cpp reactor_base.cpp)

target_include_directories(reactor PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                          $<INSTALL_INTERFACE:include>
                                   PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add the soot library
find_path(
	NASA_SOOT_DIR
	NAMES libsection_soot.so
	PATHS ${NASA_SOOT_PATH}
	)
find_library(
	NASA_SOOT_LIB
	NAMES section_soot
	PATHS ${NASA_SOOT_PATH}
)
if(NOT NASA_SOOT_DIR OR NOT NASA_SOOT_LIB)
    message(FATAL_ERROR "NASA soot library not found! Set NASA_SOOT_DIR and NASA_SOOT_LIB.")
endif()
message(WARNING "Found NASA_SOOT_DIR: ${NASA_SOOT_DIR} and NASA_SOOT_LIB: ${NASA_SOOT_LIB}")
target_link_libraries(reactor PUBLIC zerork ${NASA_SOOT_LIB})
target_include_directories(reactor PUBLIC ${NASA_SOOT_DIR})

set(public_headers const_pressure_reactor.h counterflow_reactor.h
reactor_base.h reactor_constants.h
variable_volume_reactor.h)

set_target_properties(reactor PROPERTIES
 PUBLIC_HEADER  "${public_headers}")
install(TARGETS reactor
 LIBRARY DESTINATION lib
 ARCHIVE DESTINATION lib
 PUBLIC_HEADER DESTINATION include/reactor)
