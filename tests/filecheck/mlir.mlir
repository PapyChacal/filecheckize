// RUN: filecheckize %s --strip-comments | filecheck %s --match-full-lines --check-prefix STRIP
// RUN: filecheckize %s --strip-comments --check-empty-lines | filecheck %s --check-prefix WITH-EMPTY --match-full-lines
// RUN: filecheckize %s --strip-comments --mlir-anonymize | filecheck %s --check-prefix ANONYMIZE --match-full-lines
// RUN: filecheckize %s --strip-comments --xdsl-anonymize | filecheck %s --check-prefix XDSL-ANONYMIZE --match-full-lines
// RUN: filecheckize %s --strip-comments --comments-prefix # | filecheck %s --check-prefix SHARP --match-full-lines

func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
    %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
    %name = arith.constant : i32
    %other_name = arith.constant : i32
    %2 = arith.addi %name, %other_name : i32
    func.return %1 : !test.type<"int">
}

// STRIP:      // CHECK:      func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
// STRIP-NEXT: // CHECK-NEXT:     %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
// STRIP-NEXT: // CHECK-NEXT:     %name = arith.constant : i32
// STRIP-NEXT: // CHECK-NEXT:     %other_name = arith.constant : i32
// STRIP-NEXT: // CHECK-NEXT:     %2 = arith.addi %name, %other_name : i32
// STRIP-NEXT: // CHECK-NEXT:     func.return %1 : !test.type<"int">
// STRIP-NEXT: // CHECK-NEXT: }

// WITH-EMPTY:       // CHECK-EMPTY:
// WITH-EMPTY-NEXT:  // CHECK-NEXT:  func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
// WITH-EMPTY-NEXT:  // CHECK-NEXT:      %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
// WITH-EMPTY-NEXT:  // CHECK-NEXT:      %name = arith.constant : i32
// WITH-EMPTY-NEXT:  // CHECK-NEXT:      %other_name = arith.constant : i32
// WITH-EMPTY-NEXT:  // CHECK-NEXT:      %2 = arith.addi %name, %other_name : i32
// WITH-EMPTY-NEXT:  // CHECK-NEXT:      func.return %1 : !test.type<"int">
// WITH-EMPTY-NEXT:  // CHECK-NEXT:  }
// WITH-EMPTY-NEXT:  // CHECK-EMPTY:
// WITH-EMPTY-NEXT:  // CHECK-EMPTY:
// WITH-EMPTY-NEXT:  // CHECK-EMPTY:
// WITH-EMPTY-NEXT:  // CHECK-EMPTY:

// ANONYMIZE:       // CHECK:       func.func @arg_rec(%{{.*}} : !test.type<"int">) -> !test.type<"int"> {
// ANONYMIZE-NEXT:  // CHECK-NEXT:      %{{.*}} = func.call @arg_rec(%{{.*}}) : (!test.type<"int">) -> !test.type<"int">
// ANONYMIZE-NEXT:  // CHECK-NEXT:      %{{.*}} = arith.constant : i32
// ANONYMIZE-NEXT:  // CHECK-NEXT:      %{{.*}} = arith.constant : i32
// ANONYMIZE-NEXT:  // CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// ANONYMIZE-NEXT:  // CHECK-NEXT:      func.return %{{.*}} : !test.type<"int">
// ANONYMIZE-NEXT:  // CHECK-NEXT:  }

// XDSL-ANONYMIZE:       // CHECK:       func.func @arg_rec(%{{.*}} : !test.type<"int">) -> !test.type<"int"> {
// XDSL-ANONYMIZE-NEXT:  // CHECK-NEXT:      %{{.*}} = func.call @arg_rec(%{{.*}}) : (!test.type<"int">) -> !test.type<"int">
// XDSL-ANONYMIZE-NEXT:  // CHECK-NEXT:      %name = arith.constant : i32
// XDSL-ANONYMIZE-NEXT:  // CHECK-NEXT:      %other_name = arith.constant : i32
// XDSL-ANONYMIZE-NEXT:  // CHECK-NEXT:      %{{.*}} = arith.addi %name, %other_name : i32
// XDSL-ANONYMIZE-NEXT:  // CHECK-NEXT:      func.return %{{.*}} : !test.type<"int">
// XDSL-ANONYMIZE-NEXT:  // CHECK-NEXT:  }

// SHARP:       # CHECK:       func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
// SHARP-NEXT:  # CHECK-NEXT:      %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
// SHARP-NEXT:  # CHECK-NEXT:      %name = arith.constant : i32
// SHARP-NEXT:  # CHECK-NEXT:      %other_name = arith.constant : i32
// SHARP-NEXT:  # CHECK-NEXT:      %2 = arith.addi %name, %other_name : i32
// SHARP-NEXT:  # CHECK-NEXT:      func.return %1 : !test.type<"int">
// SHARP-NEXT:  # CHECK-NEXT:  }
