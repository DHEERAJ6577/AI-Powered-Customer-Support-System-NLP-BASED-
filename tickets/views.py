from django.shortcuts import render, redirect
from .models import Ticket
from .ai_helper import generate_solution
from .classifier import predict_category   # import your classifier
from .serializer import TicketSerializer

class TicketViewSet(viewsets.ModelViewSet):
    queryset = Ticket.objects.all().order_by('-created_at')
    serializer_class = TicketSerializer
    
def create_ticket(request):
    if request.method == "POST":
        title = request.POST["title"]
        description = request.POST["description"]

        # 🔥 Predict category dynamically using BERT
        category = predict_category(description)

        # Save ticket
        ticket = Ticket.objects.create(
            title=title,
            description=description,
            customer=request.user
        )

        # Call Gemini API with predicted category
        solution = generate_solution(description, category)
        ticket.solution = solution
        ticket.save()

        return redirect("ticket_list")

    return render(request, "tickets/create_ticket.html")
