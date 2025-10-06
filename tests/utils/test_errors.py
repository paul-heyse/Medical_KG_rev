from Medical_KG_rev.utils.errors import FoundationError, ProblemDetail


def test_problem_detail_to_response():
    problem = ProblemDetail(title="Error", status=400, detail="Bad")
    response = problem.to_response()
    assert response["title"] == "Error"


def test_foundation_error_wraps_problem():
    error = FoundationError("Oops", status=404)
    assert error.problem.status == 404
