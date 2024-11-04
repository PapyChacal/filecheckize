// RUN: filecheckize %s | filecheck %s
// RUN: filecheckize --compact-output %s | filecheck %s --check-prefix COMPACT

someline
someotherline

someafterskipline
someotherafterskipline


someafter2lines
someotherafter2lines



someafter3lines
someotherafter3lines

// CHECK:       // CHECK:       someline
// CHECK-NEXT:  // CHECK-NEXT:  someotherline
// CHECK-EMPTY:
// CHECK-NEXT:  // CHECK:       someafterskipline
// CHECK-NEXT:  // CHECK-NEXT:  someotherafterskipline
// CHECK-EMPTY:
// CHECK-NEXT:  // CHECK:       someafter2lines
// CHECK-NEXT:  // CHECK-NEXT:  someotherafter2lines
// CHECK-EMPTY:
// CHECK-NEXT:  // CHECK:       someafter3lines
// CHECK-NEXT:  // CHECK-NEXT:  someotherafter3lines

// COMPACT:       // CHECK:       someline
// COMPACT-NEXT:  // CHECK-NEXT:  someotherline
// COMPACT-NEXT:  // CHECK:       someafterskipline
// COMPACT-NEXT:  // CHECK-NEXT:  someotherafterskipline
// COMPACT-NEXT:  // CHECK:       someafter2lines
// COMPACT-NEXT:  // CHECK-NEXT:  someotherafter2lines
// COMPACT-NEXT:  // CHECK:       someafter3lines
// COMPACT-NEXT:  // CHECK-NEXT:  someotherafter3lines