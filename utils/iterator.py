import itertools as it
import torch

class BatchGenerator:
    def __init__(self, source_loader, target_loader, epochs, iterations, starting_epoch=0):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iterations = len(source_loader)
        self.target_iterations = len(target_loader)
        self.cur_epoch = starting_epoch

        self.iterations = self.source_iterations if self.source_iterations < self.target_iterations else self.target_iterations
        print ("print iterations: ", self.iterations, " - ", iterations)
        
        # epoch interval that a source refresh
        self.source_to_refresh = self.source_iterations // self.iterations
        self.target_to_refresh = self.target_iterations // self.iterations

        print ("print source_iterations: ", self.source_iterations)
        print ("print target_iterations: ", self.target_iterations)
        print ("source refresh every", self.source_to_refresh, "epochs")
        print ("target refresh every", self.target_to_refresh, "epochs")

        if starting_epoch == 0:
            self.startFrom = 0    
            self.source_true_epoch = 0
            self.target_true_epoch = 0
            self.source_loader.sampler.set_epoch(self.source_true_epoch)
            self.target_loader.sampler.set_epoch(self.target_true_epoch)
            self.source_iter = iter(self.source_loader)
            self.target_iter = iter(self.target_loader)
            self.source_next_epoch = self.source_to_refresh
            self.target_next_epoch = self.target_to_refresh
            print("start from beginning")
        else:
            self.startFrom = starting_epoch * self.iterations
            source_consume = self.startFrom % self.source_iterations
            self.source_true_epoch = self.startFrom // self.source_iterations
            print("source_consume: ", source_consume)
            print("source_true_epoch: ",  self.source_true_epoch)
            target_consume = self.startFrom % self.target_iterations
            self.target_true_epoch = self.startFrom // self.target_iterations
            print("target_consume: ", target_consume)
            print("target_true_epoch: ", self.target_true_epoch)
            
            source_loader.sampler.set_epoch(self.source_true_epoch)
            target_loader.sampler.set_epoch(self.target_true_epoch)
            self.source_iter = iter(self.source_loader)
            self.target_iter = iter(self.target_loader)
            
            self.source_next_epoch = (starting_epoch // self.source_to_refresh) * (self.source_to_refresh+1)
            self.target_next_epoch = (starting_epoch // self.target_to_refresh) * (self.target_to_refresh+1)            
            
            if source_consume > 0:
                it.islice(self.source_iter, source_consume, None)
            if target_consume > 0:
                it.islice(self.target_iter, target_consume, None)
                 
    def get_batch(self):
        if self.cur_epoch != 0:
            # refresh source iterator
            if self.cur_epoch == self.source_next_epoch:
                self.source_true_epoch +=1
                self.source_next_epoch += self.source_to_refresh
                self.source_loader.sampler.set_epoch(self.source_true_epoch)
                self.source_iter = iter(self.source_loader)
                print("Reloaded source iterator, ( ep ",self.cur_epoch," )")
            
            if self.cur_epoch == self.target_next_epoch:
                self.target_true_epoch += 1
                self.target_next_epoch += self.target_to_refresh
                self.target_loader.sampler.set_epoch(self.source_true_epoch)
                self.target_iter = iter(self.target_loader)
                print("Reloaded target iterator, ( curep ",self.cur_epoch," ) , refresh (", self.source_to_refresh," ",self.target_to_refresh,")")

        self.cur_epoch += 1
        for i in range (self.iterations):
            if i % 100 == 0:
                print("Iteration:",i)
            yield i, next(self.source_iter), next(self.target_iter)


class BatchGeneratorSkipping:
    def __init__(self, source_loader, target_loader, epochs, iterations, starting_epoch=0):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iterations = len(source_loader)
        self.target_iterations = len(target_loader)
        self.cur_epoch = starting_epoch

        self.iterations = self.source_iterations if self.source_iterations < self.target_iterations else self.target_iterations
                 
    def get_batch(self):
        self.source_loader.sampler.set_epoch(self.cur_epoch)
        self.target_loader.sampler.set_epoch(self.cur_epoch)
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)

        self.cur_epoch += 1
        skip = 0
        for i in range (self.iterations):
            if i % 100 == 0:
                print("Iteration:",i)
            source_sample = next(self.source_iter)
            target_sample = next(self.target_iter)
            while skip < 5:
                if (19 in torch.unique(source_sample[1])):
                    skip = 5
                else:
                    skip += 1
                    source_sample = next(self.source_iter)
            skip = 0
            yield i, source_sample, target_sample


                
