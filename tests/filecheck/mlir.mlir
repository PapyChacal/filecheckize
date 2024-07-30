// RUN: filecheckize %s --strip-comments | filecheck %s --match-full-lines --check-prefix STRIP
// RUN: filecheckize %s --strip-comments --check-empty-lines | filecheck %s --check-prefix WITH-EMPTY --match-full-lines
// RUN: filecheckize %s --strip-comments --mlir-anonymize | filecheck %s --check-prefix MLIR-ANONYMIZE --match-full-lines
// RUN: filecheckize %s --strip-comments --mlir-anonymize --substitute | filecheck %s --check-prefix MLIR-SUBSTITUTE --match-full-lines
// RUN: filecheckize %s --strip-comments --xdsl-anonymize | filecheck %s --check-prefix XDSL-ANONYMIZE --match-full-lines
// RUN: filecheckize %s --strip-comments --xdsl-anonymize --substitute | filecheck %s --check-prefix XDSL-SUBSTITUTE --match-full-lines

func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
    %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
    %name = arith.constant : i32
    %other_name = arith.constant : i32
    %2 = arith.addi %name, %other_name : i32
    func.return %1 : !test.type<"int">
}

// STRIP{LITERAL}:      // CHECK:      func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
// STRIP-NEXT{LITERAL}: // CHECK-NEXT:     %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
// STRIP-NEXT{LITERAL}: // CHECK-NEXT:     %name = arith.constant : i32
// STRIP-NEXT{LITERAL}: // CHECK-NEXT:     %other_name = arith.constant : i32
// STRIP-NEXT{LITERAL}: // CHECK-NEXT:     %2 = arith.addi %name, %other_name : i32
// STRIP-NEXT{LITERAL}: // CHECK-NEXT:     func.return %1 : !test.type<"int">
// STRIP-NEXT{LITERAL}: // CHECK-NEXT: }

// WITH-EMPTY{LITERAL}:       // CHECK-EMPTY:
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:  func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:      %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:      %name = arith.constant : i32
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:      %other_name = arith.constant : i32
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:      %2 = arith.addi %name, %other_name : i32
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:      func.return %1 : !test.type<"int">
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-NEXT:  }
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-EMPTY:
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-EMPTY:
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-EMPTY:
// WITH-EMPTY-NEXT{LITERAL}:  // CHECK-EMPTY:

// MLIR-ANONYMIZE{LITERAL}:       // CHECK:       func.func @arg_rec(%{{.*}} : !test.type<"int">) -> !test.type<"int"> {
// MLIR-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %{{.*}} = func.call @arg_rec(%{{.*}}) : (!test.type<"int">) -> !test.type<"int">
// MLIR-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %{{.*}} = arith.constant : i32
// MLIR-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %{{.*}} = arith.constant : i32
// MLIR-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// MLIR-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      func.return %{{.*}} : !test.type<"int">
// MLIR-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:  }

// MLIR-SUBSTITUTE{LITERAL}:       // CHECK:       func.func @arg_rec([[v0:%.*]] : !test.type<"int">) -> !test.type<"int"> {
// MLIR-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      [[v1:%.*]] = func.call @arg_rec([[v0]]) : (!test.type<"int">) -> !test.type<"int">
// MLIR-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      [[vname:%.*]] = arith.constant : i32
// MLIR-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      [[vother_name:%.*]] = arith.constant : i32
// MLIR-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      [[v2:%.*]] = arith.addi [[vname]], [[vother_name]] : i32
// MLIR-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      func.return [[v1]] : !test.type<"int">
// MLIR-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:  }

// XDSL-ANONYMIZE{LITERAL}:       // CHECK:       func.func @arg_rec(%{{.*}} : !test.type<"int">) -> !test.type<"int"> {
// XDSL-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %{{.*}} = func.call @arg_rec(%{{.*}}) : (!test.type<"int">) -> !test.type<"int">
// XDSL-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %name = arith.constant : i32
// XDSL-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %other_name = arith.constant : i32
// XDSL-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      %{{.*}} = arith.addi %name, %other_name : i32
// XDSL-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:      func.return %{{.*}} : !test.type<"int">
// XDSL-ANONYMIZE-NEXT{LITERAL}:  // CHECK-NEXT:  }

// XDSL-SUBSTITUTE{LITERAL}:       // CHECK:       func.func @arg_rec([[v0:%.*]] : !test.type<"int">) -> !test.type<"int"> {
// XDSL-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      [[v1:%.*]] = func.call @arg_rec([[v0]]) : (!test.type<"int">) -> !test.type<"int">
// XDSL-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      %name = arith.constant : i32
// XDSL-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      %other_name = arith.constant : i32
// XDSL-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      [[v2:%.*]] = arith.addi %name, %other_name : i32
// XDSL-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:      func.return [[v1]] : !test.type<"int">
// XDSL-SUBSTITUTE-NEXT{LITERAL}:  // CHECK-NEXT:  }
