#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE = 1 # for asprintf
#endif
#include <stdio.h>

#include "unit-test-framework.h"

/* A simple unit test framework.  Returns NULL if there is no assertion
 * failure, and otherwise a char* with an error message.
 */
char* assert_that(bool assertion, char* assertion_message) {
  if (!assertion) {
    char* error_msg;
    asprintf(&error_msg, "Expected that %s, but that wasn't true!", assertion_message);
    free(assertion_message);
    return error_msg;
  } else {
    return NULL;
  }
}

char* both(char* assertion_1, char* assertion_2) {
  if (NULL == assertion_1) {
    return assertion_2;
  } else if (NULL == assertion_2) {
    return assertion_1;
  } else {
    char* new_error_message;
    asprintf(&new_error_message, "%s\n[and] %s", assertion_1, assertion_2);
    free(assertion_1);
    free(assertion_2);
    return new_error_message;
  }
}

test_definition* test(char* test_name, test_function* test_func) {
  test_definition* t = malloc(sizeof(test_definition));
  t->test_name = test_name;
  t->test_func = test_func;
  return t;
}

void run_tests(bool verbose, test_definition** tests) {
  int num_failures = 0;
  int num_successes = 0;
  for(test_definition** def = tests; NULL != *def; def++) {
    test_definition* test_handle = *def;
    char* test_name = test_handle->test_name;
    test_function* test_func = test_handle->test_func;
    if (verbose) {
      printf("Starting test \"%s\".\n", test_name);
    }
    char* error_msg = test_func();
    if (NULL != error_msg) {
      printf("Test \"%s\" failed with message \"%s\".\n", test_name, error_msg);
      free(error_msg);
      num_failures++;
    } else {
      num_successes++;
    }
    free(test_handle);
  }
  if (num_failures > 0) {
    printf("Some tests failed!  There were %d successes and %d failures.\n", num_successes, num_failures);
  } else {
    printf("All %d tests passed.\n", num_successes);
  }
}