
file(
  GLOB
  tobetested
  example-*.cpp
  alain-cartpole-limits.cpp
  alain-cartpole-002.cpp
)

foreach(f ${tobetested})
  get_filename_component(testName ${f} NAME_WE)
  add_executable            (${testName}                            ${f}                     )
  set_target_properties     (${testName} PROPERTIES LINKER_LANGUAGE CXX                      )
  set_target_properties     (${testName} PROPERTIES COMPILE_FLAGS   "${PROJECT_CFLAGS} -Wall")
  
  target_include_directories(${testName} PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_include_directories(${testName} PUBLIC ${GDYN_INCLUDE_DIRS})
  target_include_directories(${testName} PUBLIC ${EIGEN3_INCLUDE_DIRS})

  target_link_libraries (${testName} PUBLIC ${GDYN_LIBRARIES})
  target_link_libraries (${testName} PUBLIC ${EIGEN3_LIBRARIES})
endforeach(f)


#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#message("Eigen is in ${EIGEN3_INCLUDE_DIR}")
