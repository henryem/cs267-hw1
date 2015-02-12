#ifndef _unit_test_framework_h
#define _unit_test_framework_h

char* assert_that(bool assertion, char* assertion_message);

char* both(char* assertion_1, char* assertion_2);

typedef char* test_function();

typedef struct test_definition {
  char* test_name;
  test_function* test_func;
} test_definition;

test_definition* test(char* test_name, test_function* test_func);

/* Run all the tests.  @tests is null-terminated. */
void run_tests(bool verbose, test_definition** tests);

#endif